use std::hash::{Hash, Hasher};

use crate::{Db, FxOrderSet, ProgramEnvironment};
use ruff_python_ast as ast;
use ruff_python_ast::name::Name;
use rustc_hash::FxHashMap;
use ty_python_core::Truthiness;
use ty_python_core::predicate::{
    ClassPatternPredicateKind, MappingPatternPredicateKind, PatternPredicateKind,
    SequencePatternPredicateKind,
};

use crate::place::{DefinedPlace, Place};
use crate::types::callable::CallableTypeKind;
use crate::types::equality::{
    ComparisonSoundnessPolicy, enum_has_distinct_member_equality, evaluate_type_equality,
    is_same_enum_domain,
};
use crate::types::set_theoretic::RecursivelyDefined;
use crate::types::signatures::CallableSignature;
use crate::types::tuple::TupleType;
use crate::types::visitor::any_over_type;
use crate::types::{
    CallableType, ClassBase, ClassLiteral, EnumLiteralType, IntersectionBuilder, IntersectionType,
    KnownClass, MemberLookupPolicy, Parameter, Parameters, Signature, SpecialFormType, Type,
    TypeContext, TypeVarBoundOrConstraints, TypedDictType, UnionType, binding_type,
    equality_truthiness, infer_same_file_expression_type,
};

/// Identifies a pattern by its address and subject type without hashing the pattern subtree.
#[derive(Clone, Copy)]
pub(super) struct PatternCacheKey<'pattern, 'db> {
    pub(super) pattern: &'pattern PatternPredicateKind<'db>,
    pub(super) subject_ty: Type<'db>,
}

impl PatternCacheKey<'_, '_> {
    pub(super) fn is_structural(self) -> bool {
        matches!(
            self.pattern,
            PatternPredicateKind::Class(_)
                | PatternPredicateKind::Mapping(_)
                | PatternPredicateKind::Sequence(_)
                | PatternPredicateKind::Or(_)
        )
    }
}

impl PartialEq for PatternCacheKey<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.pattern, other.pattern) && self.subject_ty == other.subject_ty
    }
}

impl Eq for PatternCacheKey<'_, '_> {}

impl Hash for PatternCacheKey<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self.pattern, state);
        self.subject_ty.hash(state);
    }
}

type PatternTypeCache<'pattern, 'db> = FxHashMap<PatternCacheKey<'pattern, 'db>, Type<'db>>;

pub(crate) fn singleton_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    singleton: ast::Singleton,
) -> Type<'db> {
    let ty = match singleton {
        ast::Singleton::None => Type::none(db, env),
        ast::Singleton::True => Type::bool_literal(true),
        ast::Singleton::False => Type::bool_literal(false),
    };
    debug_assert!(ty.is_singleton(db, env));
    ty
}

pub(crate) fn mapping_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
) -> Type<'db> {
    KnownClass::Mapping
        .to_instance(db, env)
        .top_materialization(db, env)
}

/// Return the class expression type for coverage analysis.
pub(crate) fn class_pattern_coverage_target<'db>(
    db: &'db dyn Db,
    pattern: &ClassPatternPredicateKind<'db>,
) -> Type<'db> {
    let class_ty = infer_same_file_expression_type(db, pattern.class, TypeContext::default());
    // After `cls is A`, the type of `A` can be `<class 'A'> & type[T]`. The class literal
    // still fixes the runtime class that the pattern tests, so coverage can use it directly.
    match class_ty {
        Type::Intersection(intersection) => intersection
            .positive(db)
            .iter()
            .copied()
            .find(|ty| matches!(ty, Type::ClassLiteral(_)))
            .unwrap_or(class_ty),
        _ => class_ty,
    }
}

/// Return whether every runtime value represented by a `TypedDict` satisfies `class`.
///
/// `TypedDict` is not a nominal subtype of `dict` in the static type system, but every runtime
/// value is a dictionary. A `TypedDict` therefore matches class patterns such as `dict()`,
/// `Mapping()`, and `MutableMapping()`.
pub(crate) fn typed_dict_matches_class_pattern<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    class: ClassLiteral<'db>,
) -> bool {
    let Some(dict) = KnownClass::Dict
        .to_class_literal(db, env)
        .as_class_literal()
    else {
        return false;
    };
    Type::instance(db, env, dict.top_materialization(db)).is_subtype_of(
        db,
        env,
        Type::instance(db, env, class.top_materialization(db)),
    )
}

/// Return whether every value in `ty` belongs to a `TypedDict` domain accepted by `predicate`.
fn typed_dict_pattern_domain_satisfies<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    ty: Type<'db>,
    predicate: &mut impl FnMut(TypedDictType<'db>) -> bool,
) -> bool {
    match ty.resolve_type_alias(db) {
        Type::TypedDict(typed_dict) => predicate(typed_dict),
        Type::TypeVar(typevar) => match typevar.typevar(db).bound_or_constraints(db, env) {
            Some(TypeVarBoundOrConstraints::UpperBound(bound)) => {
                typed_dict_pattern_domain_satisfies(db, env, bound, predicate)
            }
            Some(TypeVarBoundOrConstraints::Constraints(constraints)) => {
                constraints.elements(db).iter().all(|constraint| {
                    typed_dict_pattern_domain_satisfies(db, env, *constraint, predicate)
                })
            }
            None => false,
        },
        Type::Union(union) => union
            .elements(db)
            .iter()
            .all(|element| typed_dict_pattern_domain_satisfies(db, env, *element, predicate)),
        Type::Intersection(intersection) => intersection
            .positive(db)
            .iter()
            .any(|element| typed_dict_pattern_domain_satisfies(db, env, *element, predicate)),
        _ => false,
    }
}

/// Return whether every value in `ty` is represented by a `TypedDict` schema at runtime.
pub(super) fn is_typed_dict_runtime_domain(
    db: &dyn Db,
    env: &ProgramEnvironment<'_>,
    ty: Type<'_>,
) -> bool {
    typed_dict_pattern_domain_satisfies(db, env, ty, &mut |_| true)
}

pub(crate) fn sequence_pattern_type_builder<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
) -> IntersectionBuilder<'db> {
    IntersectionBuilder::new(db, env)
        .add_positive(
            KnownClass::Sequence
                .to_instance(db, env)
                .top_materialization(db, env),
        )
        // `str`, `bytes`, and `bytearray` are sequences, but Python sequence
        // patterns explicitly do not match them or their subclasses.
        .add_negative(KnownClass::Str.to_instance(db, env))
        .add_negative(KnownClass::Bytes.to_instance(db, env))
        .add_negative(KnownClass::Bytearray.to_instance(db, env))
}

fn sequence_pattern_getitem_method<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    indexed_element_types: impl IntoIterator<Item = (i64, Type<'db>)>,
    fallback_return_type: Option<Type<'db>>,
) -> CallableType<'db> {
    let self_parameter = || Parameter::positional_only(Some(Name::new_static("self")));

    let overloads = indexed_element_types
        .into_iter()
        .map(|(index, element_type)| {
            Signature::new(
                Parameters::standard([
                    self_parameter(),
                    Parameter::positional_only(Some(Name::new_static("index")))
                        .with_annotated_type(Type::int_literal(index)),
                ]),
                element_type,
            )
        });
    let fallback_overload = fallback_return_type.map(|fallback_return_type| {
        Signature::new(
            Parameters::standard([
                self_parameter(),
                Parameter::positional_only(Some(Name::new_static("index")))
                    .with_annotated_type(KnownClass::Int.to_instance(db, env)),
            ]),
            fallback_return_type,
        )
    });

    CallableType::new(
        db,
        CallableSignature::from_overloads(overloads.chain(fallback_overload)),
        CallableTypeKind::FunctionLike,
    )
}

/// Build the structural type used for a fixed-length sequence pattern.
///
/// For a pattern like:
///
/// ```python
/// match value:
///     case [int(), str()]:
///         ...
/// ```
///
/// this returns the sequence-pattern runtime type plus a synthesized protocol
/// whose `__len__` and indexed `__getitem__` methods encode the fixed length
/// and element types.
pub(crate) fn exact_sequence_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    element_types: impl ExactSizeIterator<Item = Type<'db>>,
) -> Type<'db> {
    let Ok(length) = i64::try_from(element_types.len()) else {
        return sequence_pattern_type_builder(db, env).build();
    };

    // `False == 0` and `True == 1`, so the protocol must accept both literals.
    let length_type = match length {
        0 => UnionType::from_two_elements(db, env, Type::int_literal(0), Type::bool_literal(false)),
        1 => UnionType::from_two_elements(db, env, Type::int_literal(1), Type::bool_literal(true)),
        _ => Type::int_literal(length),
    };

    let self_parameter = || Parameter::positional_only(Some(Name::new_static("self")));

    let len_signature = Signature::new(Parameters::standard([self_parameter()]), length_type);
    let len_method = CallableType::function_like(db, len_signature);

    let getitem_method = (element_types.len() > 0).then(|| {
        (
            "__getitem__",
            sequence_pattern_getitem_method(db, env, (0..length).zip(element_types), None),
        )
    });

    let protocol = Type::protocol_with_methods(
        db,
        env,
        std::iter::once(("__len__", len_method)).chain(getitem_method),
    );

    sequence_pattern_type_builder(db, env)
        .add_positive(protocol)
        .build()
}

/// Build the structural type used for a sequence pattern containing `*rest`.
///
/// Fixed prefix elements use non-negative indices and fixed suffix elements use
/// negative indices. Other integer indices retain the sequence's element type.
pub(crate) fn starred_sequence_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    prefix_element_types: impl ExactSizeIterator<Item = Type<'db>>,
    suffix_element_types: impl ExactSizeIterator<Item = Type<'db>>,
) -> Type<'db> {
    if prefix_element_types.len() == 0 && suffix_element_types.len() == 0 {
        return sequence_pattern_type_builder(db, env).build();
    }

    let Ok(suffix_length) = i64::try_from(suffix_element_types.len()) else {
        return sequence_pattern_type_builder(db, env).build();
    };

    let indexed_element_types = (0_i64..)
        .zip(prefix_element_types)
        .chain((-suffix_length..0).zip(suffix_element_types));
    let getitem_method =
        sequence_pattern_getitem_method(db, env, indexed_element_types, Some(Type::object()));
    let protocol = Type::protocol_with_methods(db, env, [("__getitem__", getitem_method)]);

    sequence_pattern_type_builder(db, env)
        .add_positive(protocol)
        .build()
}

/// Return whether every value in `subject_ty` is statically guaranteed to match this class pattern.
///
/// Attribute subpatterns are checked recursively against their statically known member types. The
/// subject's class can provide members that are absent from the class named in the pattern.
///
/// ```python
/// class Base: ...
///
/// class Child(Base):
///     x: int
///
/// # Exhaustive for a `Child` subject because `Child.x` is definitely bound.
/// case Base(x=_): ...
/// ```
fn class_pattern_is_exhaustive<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    class: ClassLiteral<'db>,
    subject_ty: Type<'db>,
    kind: &'pattern ClassPatternPredicateKind<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> bool {
    let class_instance_ty = Type::instance(db, env, class.top_materialization(db));
    let is_typed_dict_match = is_typed_dict_runtime_domain(db, env, subject_ty)
        && typed_dict_matches_class_pattern(db, env, class);
    if !is_typed_dict_match && !subject_ty.is_subtype_of(db, env, class_instance_ty) {
        return false;
    }

    if kind.is_empty() {
        return true;
    }

    if !kind.keywords.iter().all(|keyword| {
        member_pattern_is_exhaustive(
            db,
            env,
            subject_ty,
            keyword.attr.as_str(),
            &keyword.pattern,
            cache,
        )
    }) {
        return false;
    }

    let positional_sources =
        class_pattern_positional_sources(db, env, class, kind.positional.len());
    kind.positional
        .iter()
        .zip(positional_sources)
        .all(|(pattern, source)| match source {
            ClassPatternPositionalSource::MatchSelf => {
                pattern_is_exhaustive_for_subject(db, env, pattern, subject_ty, cache)
            }
            ClassPatternPositionalSource::Attribute(name) => {
                member_pattern_is_exhaustive(db, env, subject_ty, name.as_str(), pattern, cache)
            }
            ClassPatternPositionalSource::Unknown => false,
        })
}

/// The static lookup result for a pattern class's `__match_args__` member.
///
/// `PossiblyUndefined` is distinct from `Undefined` because a conditional definition can take
/// precedence over match-self behavior at runtime, so that behavior cannot be assumed.
enum ClassMatchArgs<'db> {
    /// The class and its metaclass hierarchy do not define `__match_args__`.
    Undefined,
    /// `__match_args__` is definitely bound.
    Defined {
        /// The semantic member type used to validate positional subpatterns.
        member_type: Type<'db>,
        /// The type used to resolve positional attribute names.
        positional_source_type: Type<'db>,
    },
    /// `__match_args__` is defined along some control-flow paths but not others.
    PossiblyUndefined,
}

/// The value supplied to one positional subpattern in a class pattern.
#[derive(Clone)]
pub(crate) enum ClassPatternPositionalSource {
    /// The complete subject, as used by Python's special built-in class patterns.
    MatchSelf,
    /// The named attribute extracted from the subject according to `__match_args__`.
    Attribute(Name),
    /// No value source can be determined statically, so the subpattern is not exhaustive.
    Unknown,
}

/// Resolve `__match_args__` through the pattern class, including its metaclass.
///
/// The semantic member type is used for validation. When mapping positional subpatterns to
/// attributes, inferred assignments retain their literal binding type while an explicit annotation
/// remains authoritative. `PossiblyUndefined` is distinct from `Undefined` because only a truly
/// absent `__match_args__` enables match-self behavior.
fn class_match_args_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    class: ClassLiteral<'db>,
) -> ClassMatchArgs<'db> {
    match Type::ClassLiteral(class)
        .member(db, env, "__match_args__")
        .place
    {
        Place::Defined(
            place @ DefinedPlace {
                ty,
                origin,
                provenance,
                ..
            },
        ) if place.is_definitely_defined() => ClassMatchArgs::Defined {
            member_type: ty,
            positional_source_type: if origin.is_declared() {
                ty
            } else {
                provenance
                    .definition()
                    .map_or(ty, |definition| binding_type(db, definition))
            },
        },
        Place::Defined(_) => ClassMatchArgs::PossiblyUndefined,
        Place::Undefined => ClassMatchArgs::Undefined,
    }
}

/// Return whether `class` inherits Python's special match-self class-pattern behavior.
///
/// Callers must first establish that `__match_args__` is statically absent. A definite definition
/// overrides match-self behavior, while a conditional definition makes it runtime-dependent;
/// neither case should consult this flag.
fn class_has_match_self_flag(db: &dyn Db, class: ClassLiteral<'_>) -> bool {
    class
        .iter_mro(db)
        .filter_map(ClassBase::into_class)
        .any(|base| {
            matches!(
                base.class_literal(db).known(db),
                Some(
                    KnownClass::Bool
                        | KnownClass::Bytearray
                        | KnownClass::Bytes
                        | KnownClass::Dict
                        | KnownClass::Float
                        | KnownClass::FrozenSet
                        | KnownClass::Int
                        | KnownClass::List
                        | KnownClass::Set
                        | KnownClass::Str
                        | KnownClass::Tuple
                )
            )
        })
}

/// The statically known result of validating positional subpatterns for a class pattern.
pub(crate) enum ClassPatternPositionalResult<'db> {
    /// The maximum number of positional subpatterns accepted by the class.
    Limit(usize),
    /// A statically known non-tuple value used for `__match_args__`.
    InvalidType(Type<'db>),
}

/// Validate positional subpatterns against a statically known `__match_args__` type.
pub(crate) fn class_pattern_positional_result<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    class: ClassLiteral<'db>,
) -> Option<ClassPatternPositionalResult<'db>> {
    match class_match_args_type(db, env, class) {
        ClassMatchArgs::Undefined if class_has_match_self_flag(db, class) => {
            Some(ClassPatternPositionalResult::Limit(1))
        }
        ClassMatchArgs::Undefined
            if class.known(db).is_some()
                || class
                    .as_static()
                    .is_some_and(|class| !class.body_scope(db).file(db).is_stub(db)) =>
        {
            Some(ClassPatternPositionalResult::Limit(0))
        }
        ClassMatchArgs::Defined { member_type, .. } => {
            let match_args = member_type.resolve_type_alias(db);
            if let Some(limit) = match_args.exact_tuple_instance_spec(db).and_then(|tuple| {
                tuple
                    .as_fixed_length()
                    .map(super::tuple::FixedLengthTuple::len)
            }) {
                Some(ClassPatternPositionalResult::Limit(limit))
            } else {
                match_args
                    .is_disjoint_from(db, env, Type::homogeneous_tuple(db, env, Type::unknown()))
                    .then_some(ClassPatternPositionalResult::InvalidType(match_args))
            }
        }
        ClassMatchArgs::Undefined | ClassMatchArgs::PossiblyUndefined => None,
    }
}

/// Resolve the value supplied to each positional subpattern, preserving source order and length.
///
/// A fixed tuple of string literals in `__match_args__` maps positions to subject attributes.
/// Python's special built-in classes map only the first position to the complete subject. Missing,
/// extra, widened, or conditionally defined positions resolve to
/// [`ClassPatternPositionalSource::Unknown`].
///
/// ```python
/// class Point:
///     __match_args__ = ("x",)
///     x: int
///
/// match point:
///     case Point(_):  # The positional subpattern receives `point.x`.
///         pass
///
/// match number:
///     case int(_):  # The positional subpattern receives `number` itself.
///         pass
/// ```
pub(crate) fn class_pattern_positional_sources(
    db: &dyn Db,
    env: &ProgramEnvironment<'_>,
    class: ClassLiteral<'_>,
    positional_count: usize,
) -> Vec<ClassPatternPositionalSource> {
    let fixed = match class_match_args_type(db, env, class) {
        ClassMatchArgs::Undefined if class_has_match_self_flag(db, class) => {
            return (0..positional_count)
                .map(|index| {
                    if index == 0 {
                        ClassPatternPositionalSource::MatchSelf
                    } else {
                        ClassPatternPositionalSource::Unknown
                    }
                })
                .collect();
        }
        ClassMatchArgs::Defined {
            positional_source_type,
            ..
        } => positional_source_type
            .exact_tuple_instance_spec(db)
            .and_then(|tuple| tuple.as_fixed_length().cloned()),
        ClassMatchArgs::Undefined | ClassMatchArgs::PossiblyUndefined => None,
    };

    (0..positional_count)
        .map(|index| {
            fixed
                .as_ref()
                .and_then(|tuple| tuple.elements_slice().get(index))
                .and_then(|ty| ty.as_string_literal())
                .map(|literal| {
                    ClassPatternPositionalSource::Attribute(Name::new(literal.value(db)))
                })
                .unwrap_or(ClassPatternPositionalSource::Unknown)
        })
        .collect()
}

/// Return whether `name` is definitely bound and `pattern` consumes its entire static member type.
fn member_pattern_is_exhaustive<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    instance_ty: Type<'db>,
    name: &str,
    pattern: &'pattern PatternPredicateKind<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> bool {
    definitely_bound_member_type(db, env, instance_ty, name, MemberLookupPolicy::default())
        .is_some_and(|member_ty| {
            pattern_is_exhaustive_for_subject(db, env, pattern, member_ty, cache)
        })
}

/// Return the type of `name` on `instance_ty` if the attribute is definitely bound.
fn definitely_bound_member_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    instance_ty: Type<'db>,
    name: &str,
    policy: MemberLookupPolicy,
) -> Option<Type<'db>> {
    let place = instance_ty
        .member_lookup_with_policy(db, env, name, policy)
        .place;
    if place.is_definitely_bound() {
        place.raw_type()
    } else {
        None
    }
}

/// Return the type of `name` on `instance_ty` if reading the attribute in a later `match` case
/// gives the same value as reading it in an earlier one.
///
/// A plain instance or class attribute keeps its value from one case to the next, because the
/// cases themselves only read attributes, check classes and compare values, which we assume have
/// no side effects. A property or another descriptor, `__getattr__`, or a custom
/// `__getattribute__` can instead return a different value on every read:
///
/// ```python
/// class Coin:
///     @property
///     def side(self) -> Literal["heads", "tails"]:
///         return random.choice(["heads", "tails"])
///
/// def flip(coin: Coin) -> str:
///     match coin:
///         case Coin(side="heads"):
///             return "heads"
///         case Coin(side="tails"):  # This can fail too, so the match is not exhaustive.
///             return "tails"
/// ```
///
/// Slot descriptors read directly from instance storage, so they are as stable as plain
/// attributes.
fn stable_member_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    instance_ty: Type<'db>,
    name: &str,
) -> Option<Type<'db>> {
    let may_change_between_reads = |receiver: Type<'db>| {
        let getattribute_policy = MemberLookupPolicy::MRO_NO_OBJECT_FALLBACK
            | MemberLookupPolicy::META_CLASS_NO_TYPE_FALLBACK;
        if !receiver
            .class_member_with_policy(db, env, "__getattribute__", getattribute_policy)
            .place
            .is_undefined()
        {
            return true;
        }
        receiver
            .class_member(db, env, name)
            .place
            .raw_type()
            .is_some_and(|class_attribute_ty| {
                !matches!(class_attribute_ty, Type::SlotDescriptor(_))
                    && !class_attribute_ty
                        .class_member(db, env, "__get__")
                        .place
                        .is_undefined()
            })
    };

    // Synthesized protocols, such as the ones that earlier cases intersect with the subject,
    // describe attribute values rather than descriptors on the runtime class. Their members are
    // read-only properties, which would otherwise look like descriptors that can change.
    let is_synthesized_protocol = |ty: &Type<'db>| {
        ty.as_protocol_instance()
            .is_some_and(|protocol| protocol.class_origin(db).is_none())
    };
    let may_change = match instance_ty {
        Type::Intersection(intersection) => intersection
            .iter_positive(db)
            .filter(|ty| !is_synthesized_protocol(ty))
            .any(may_change_between_reads),
        _ => may_change_between_reads(instance_ty),
    };
    if may_change {
        return None;
    }

    definitely_bound_member_type(
        db,
        env,
        instance_ty,
        name,
        MemberLookupPolicy::NO_GETATTR_LOOKUP,
    )
}

/// Return whether `pattern` is statically guaranteed to match every value in `subject_ty`.
fn pattern_is_exhaustive_for_subject<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    pattern: &'pattern PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> bool {
    subject_ty.is_subtype_of(
        db,
        env,
        definite_match_pattern_type_for_subject_cached(db, env, pattern, subject_ty, cache),
    )
}

/// Return whether every value in `subject_ty` is guaranteed to match a mapping pattern.
///
/// A nonempty mapping pattern is exhaustive for a `TypedDict` only when every key names a required
/// field and every nested pattern exhausts that field's declared type. Other mapping types do not
/// guarantee that a particular key is present.
fn mapping_pattern_is_exhaustive<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &'pattern MappingPatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> bool {
    typed_dict_pattern_domain_satisfies(db, env, subject_ty, &mut |typed_dict| {
        kind.entries.iter().all(|entry| {
            let key_ty = infer_same_file_expression_type(db, entry.key, TypeContext::default());
            let Some(key) = key_ty.as_string_literal() else {
                return false;
            };
            typed_dict.item(db, key.value(db)).is_some_and(|field| {
                field.is_required()
                    && pattern_is_exhaustive_for_subject(
                        db,
                        env,
                        &entry.pattern,
                        field.declared_ty,
                        cache,
                    )
            })
        })
    })
}

/// Return whether an exact tuple subject is fully consumed by a sequence pattern.
///
/// Each aligned element is checked with the subject-aware matcher so nested class patterns use the
/// tuple element's actual static type.
fn sequence_pattern_is_exhaustive_for_subject<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &'pattern SequencePatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> bool {
    if !subject_ty.is_subtype_of(db, env, sequence_pattern_type_builder(db, env).build()) {
        return false;
    }

    if kind.is_irrefutable() {
        return true;
    }

    let Some(tuple) = subject_ty.exact_tuple_instance_spec(db) else {
        return false;
    };
    let Some(tuple) = tuple.as_fixed_length() else {
        return false;
    };
    let elements = tuple.all_elements();

    let Some((prefix, suffix)) = kind.split_around_star() else {
        return elements.len() == kind.patterns.len()
            && elements
                .iter()
                .zip(kind.patterns.iter())
                .all(|(element, pattern)| {
                    pattern_is_exhaustive_for_subject(db, env, pattern, *element, cache)
                });
    };
    if elements.len() < prefix.len() + suffix.len() {
        return false;
    }

    elements
        .iter()
        .zip(prefix)
        .chain(elements.iter().rev().zip(suffix.iter().rev()))
        .all(|(element, pattern)| {
            pattern_is_exhaustive_for_subject(db, env, pattern, *element, cache)
        })
}

/// Return the values that are statically guaranteed to match `kind`, using `subject_ty` when the
/// answer depends on the subject.
///
/// This is an under-approximation used for negative narrowing and ordered alternatives: callers
/// may subtract the result from `subject_ty`. A subject-independent pattern can return a type wider
/// than `subject_ty`; for example, `case Base()` returns `Base` even for a `Child` subject. Class
/// patterns need the current subject type when member extraction depends on the subject's statically
/// known members.
/// A subject-independent pattern can return its context-free definite-match type directly. A
/// mapping pattern can use required `TypedDict` fields to establish that every subject value
/// contains its keys.
/// This treats access to a definitely bound descriptor as successful even though the descriptor
/// could raise at runtime. The same rule is propagated through nested sequence, `or`, and `as`
/// patterns.
///
/// When a pattern exhausts a gradual subject, the definite-match type is the subject's top
/// materialization. Negative narrowing can then eliminate every materialization of the subject,
/// including when exhaustiveness depends on a nested pattern.
///
/// ```python
/// class Base:
///     x: int
///
/// class Child(Base):
///     pass
///
/// # For a `tuple[Child]` subject, this checks `x` on `Child`, not only on `Base`.
/// case [Base(x=_)]: ...
/// ```
pub(crate) fn definite_match_pattern_type_for_subject<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
) -> Type<'db> {
    definite_match_pattern_type_for_subject_cached(
        db,
        env,
        kind,
        subject_ty,
        &mut FxHashMap::default(),
    )
}

/// Reuse definite-match types when different union arms reach the same nested pattern and type.
///
/// ```python
/// from __future__ import annotations
/// from typing import TypeAlias, TypedDict
///
/// class A(TypedDict):
///     child: Node
/// class B(TypedDict):
///     child: Node
/// Node: TypeAlias = A | B | None
///
/// def visit(node: Node) -> None:
///     match node:
///         case {"child": {"child": _}}: ...
/// ```
///
/// Both `A` and `B` pass `Node` to the nested mapping pattern. Its definite-match type can be
/// reused when deciding which values can reach a later case. The cache is local to one analysis,
/// where the database and program environment remain the same.
fn definite_match_pattern_type_for_subject_cached<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &'pattern PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> Type<'db> {
    let key = PatternCacheKey {
        pattern: kind,
        subject_ty,
    };
    let cacheable = key.is_structural() || matches!(kind, PatternPredicateKind::As(Some(_), _));
    if cacheable && let Some(result) = cache.get(&key) {
        return *result;
    }

    let result = definite_match_pattern_type_for_subject_impl(db, env, kind, subject_ty, cache);
    if cacheable {
        cache.insert(key, result);
    }
    result
}

fn definite_match_pattern_type_for_subject_impl<'pattern, 'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &'pattern PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    cache: &mut PatternTypeCache<'pattern, 'db>,
) -> Type<'db> {
    if let Some(subject_independent_ty) =
        subject_independent_definite_match_pattern_type(db, env, kind)
    {
        return subject_independent_ty;
    }

    let resolved_subject_ty = subject_ty.resolve_type_alias(db);
    if let Type::Union(union) = resolved_subject_ty {
        return UnionType::from_elements(
            db,
            env,
            union.elements(db).iter().map(|element| {
                definite_match_pattern_type_for_subject_cached(db, env, kind, *element, cache)
            }),
        );
    }

    match kind {
        PatternPredicateKind::Value(value) => {
            let value_ty = infer_same_file_expression_type(db, *value, TypeContext::default());
            if equality_truthiness(
                db,
                env,
                resolved_subject_ty,
                value_ty,
                ComparisonSoundnessPolicy::from_analysis_settings(
                    db.analysis_settings(value.file(db)),
                ),
            ) == Truthiness::AlwaysTrue
            {
                return subject_ty;
            }
        }
        PatternPredicateKind::Class(kind) => {
            let class_ty = class_pattern_coverage_target(db, kind);
            match class_ty {
                Type::ClassLiteral(class) => {
                    if class_pattern_is_exhaustive(db, env, class, resolved_subject_ty, kind, cache)
                    {
                        let top_subject_ty = resolved_subject_ty.top_materialization(db, env);
                        if !class_pattern_is_exhaustive(db, env, class, top_subject_ty, kind, cache)
                        {
                            return subject_ty;
                        }
                        return top_subject_ty;
                    }
                }
                Type::SpecialForm(SpecialFormType::CollectionsAbcCallable)
                    if kind.is_empty()
                        && let callable_pattern_ty = Type::Callable(CallableType::top(db))
                        && subject_ty.is_subtype_of(db, env, callable_pattern_ty) =>
                {
                    return callable_pattern_ty;
                }
                _ => {}
            }
        }
        PatternPredicateKind::Sequence(kind) => {
            if !sequence_pattern_is_exhaustive_for_subject(
                db,
                env,
                kind,
                resolved_subject_ty,
                cache,
            ) {
                // A nested subject-dependent pattern rejected the context-free approximation.
                // Reusing that approximation for the surrounding sequence would reintroduce the
                // values that the recursive analysis deliberately excluded.
                return Type::Never;
            }
            let top_subject_ty = resolved_subject_ty.top_materialization(db, env);
            return if sequence_pattern_is_exhaustive_for_subject(
                db,
                env,
                kind,
                top_subject_ty,
                cache,
            ) {
                top_subject_ty
            } else {
                subject_ty
            };
        }
        PatternPredicateKind::Mapping(kind) => {
            if !mapping_pattern_is_exhaustive(db, env, kind, resolved_subject_ty, cache) {
                return Type::Never;
            }
            let top_subject_ty = resolved_subject_ty.top_materialization(db, env);
            return if mapping_pattern_is_exhaustive(db, env, kind, top_subject_ty, cache) {
                top_subject_ty
            } else {
                subject_ty
            };
        }
        PatternPredicateKind::Or(patterns) => {
            return UnionType::from_elements(
                db,
                env,
                patterns.iter().map(|pattern| {
                    definite_match_pattern_type_for_subject_cached(
                        db, env, pattern, subject_ty, cache,
                    )
                }),
            );
        }
        PatternPredicateKind::As(Some(pattern), _) => {
            return definite_match_pattern_type_for_subject_cached(
                db, env, pattern, subject_ty, cache,
            );
        }
        _ => return Type::Never,
    }

    IntersectionBuilder::new(db, env)
        .add_positive(subject_ty)
        .add_positive(definite_match_pattern_type(db, env, kind))
        .build()
}

/// Return the part of `subject_ty` that can reach a later alternative after `kind` fails.
///
/// Value patterns use Python equality. Reusing the equality constraint here both accounts for
/// cross-type equal values and avoids checking every member of a large literal union separately:
///
/// ```python
/// from typing import Literal
///
/// def f(value: Literal[True, 1, 2]) -> None:
///     match value:
///         case 1:
///             pass
///         case other:
///             reveal_type(other)  # Literal[2]
/// ```
fn pattern_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
) -> Type<'db> {
    if let PatternPredicateKind::Value(value) = kind {
        let value_ty = infer_same_file_expression_type(db, *value, TypeContext::default());
        // A subject confined to the same enum cannot contain cross-type values that compare equal
        // to the pattern, so direct subtraction avoids repeated equality evaluation in large enum
        // matches. This includes narrowed intersections containing `Self` or another type variable
        // whose upper bound is that enum.
        if let Some(enum_literal) = value_ty.as_enum_literal()
            && is_same_enum_pattern_domain(db, env, subject_ty, enum_literal)
            && equality_truthiness(
                db,
                env,
                value_ty,
                value_ty,
                ComparisonSoundnessPolicy::from_analysis_settings(
                    db.analysis_settings(value.file(db)),
                ),
            ) == Truthiness::AlwaysTrue
        {
            return IntersectionBuilder::new(db, env)
                .add_positive(subject_ty)
                .add_negative(value_ty)
                .build();
        }
        if let Some(constraint) = evaluate_type_equality(
            db,
            env,
            subject_ty,
            value_ty,
            false,
            ComparisonSoundnessPolicy::from_analysis_settings(db.analysis_settings(value.file(db))),
        ) {
            return IntersectionBuilder::new(db, env)
                .add_positive(subject_ty)
                .add_positive(constraint)
                .build();
        }
    }

    // Reachability and capture inference both walk OR patterns in order. Caching the difference
    // lets them reuse the remaining subject type after each alternative fails.
    let definitely_matched = definite_match_pattern_type_for_subject(db, env, kind, subject_ty);
    IntersectionType::from_positive_and_negative(db, env, subject_ty, definitely_matched)
}

/// Return the exact set of values matched within the subject's enum by an OR of member patterns.
pub(crate) fn same_enum_or_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    patterns: &[PatternPredicateKind<'db>],
    subject_ty: Type<'db>,
) -> Option<Type<'db>> {
    let mut literals = patterns.iter().map(|pattern| {
        let PatternPredicateKind::Value(value) = pattern else {
            return None;
        };
        infer_same_file_expression_type(db, *value, TypeContext::default()).as_enum_literal()
    });
    let first = literals.next()??;
    let enum_class = first.enum_class_literal(db);
    if !is_same_enum_pattern_domain(db, env, subject_ty, first)
        || !enum_has_distinct_member_equality(db, enum_class)
    {
        return None;
    }

    // Expression inference already resolves aliases to canonical member literals.
    let mut members = FxOrderSet::from_iter([Type::enum_literal(first)]);
    for literal in literals {
        let literal = literal?;
        if literal.enum_class_literal(db) != enum_class {
            return None;
        }
        members.insert(Type::enum_literal(literal));
    }

    if members.len() == 1 {
        return members.first().copied();
    }
    // A general union builder can widen a large literal union to the whole enum, which would
    // incorrectly exclude the unmatched members when the pattern fails.
    let members: Box<[_]> = members.into_iter().collect();
    Some(Type::Union(UnionType::new(
        db,
        members,
        RecursivelyDefined::No,
    )))
}

/// Return the fallthrough type for a binding that can reach a later match case.
///
/// Failure of a sequence pattern establishes length and indexed-element facts at the instant of
/// matching, but those facts can become stale for mutable or stateful sequences. Exact tuples are
/// immutable, so they retain normal sequence-pattern fallthrough narrowing.
///
/// In the example below, `subject_ty` is `tuple[int | str, int | str]`, `kind` represents the
/// `[int(), str()]` sequence pattern, and the returned fallthrough type is
/// `tuple[str, int | str] | tuple[int | str, int]`.
///
/// ```python
/// def f(value: tuple[int | str, int | str]) -> None:
///     match value:
///         case [int(), str()]:
///             pass
///         case _:
///             # tuple[str, int | str] | tuple[int | str, int]
///             reveal_type(value)
/// ```
pub(crate) fn pattern_binding_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
) -> Type<'db> {
    let mut budget = PatternExpansionBudget::default();
    try_pattern_binding_fallthrough_type(db, env, kind, subject_ty, &mut budget).unwrap_or_else(
        |()| conservative_pattern_binding_fallthrough_type(db, env, kind, subject_ty),
    )
}

/// Compute binding fallthrough while charging every nested exact-tuple expansion to `budget`.
///
/// An error means that the caller must discard the partially expanded type and recompute the
/// complete pattern conservatively.
fn try_pattern_binding_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    budget: &mut PatternExpansionBudget,
) -> Result<Type<'db>, ()> {
    match kind {
        PatternPredicateKind::Sequence(sequence) => {
            try_sequence_pattern_binding_fallthrough_type(db, env, sequence, subject_ty, budget)
        }
        PatternPredicateKind::Class(class) => {
            try_class_pattern_binding_fallthrough_type(db, env, kind, class, subject_ty, budget)
        }
        PatternPredicateKind::Or(patterns)
            if let Some(matched) = same_enum_or_pattern_type(db, env, patterns, subject_ty) =>
        {
            // Subtract all the members in one intersection so we don't rebuild the growing list
            // of exclusions for every alternative, especially when the subject is `Self`.
            Ok(IntersectionType::from_positive_and_negative(
                db, env, subject_ty, matched,
            ))
        }
        PatternPredicateKind::Or(patterns) => {
            patterns.iter().try_fold(subject_ty, |remaining, pattern| {
                try_pattern_binding_fallthrough_type(db, env, pattern, remaining, budget)
            })
        }
        PatternPredicateKind::As(Some(pattern), _) => {
            try_pattern_binding_fallthrough_type(db, env, pattern, subject_ty, budget)
        }
        _ => Ok(pattern_fallthrough_type(db, env, kind, subject_ty)),
    }
}

/// Compute binding fallthrough without expanding exact tuples.
///
/// This preserves the recursive handling of `Or` and `As` patterns while providing the fallback
/// used when the precise traversal exceeds its expansion budget.
fn conservative_pattern_binding_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
    subject_ty: Type<'db>,
) -> Type<'db> {
    match kind {
        PatternPredicateKind::Or(patterns)
            if let Some(matched) = same_enum_or_pattern_type(db, env, patterns, subject_ty) =>
        {
            IntersectionType::from_positive_and_negative(db, env, subject_ty, matched)
        }
        PatternPredicateKind::Or(patterns) => {
            patterns.iter().fold(subject_ty, |remaining, pattern| {
                conservative_pattern_binding_fallthrough_type(db, env, pattern, remaining)
            })
        }
        PatternPredicateKind::As(Some(pattern), _) => {
            conservative_pattern_binding_fallthrough_type(db, env, pattern, subject_ty)
        }
        _ => pattern_fallthrough_type(db, env, kind, subject_ty),
    }
}

/// Apply sequence-pattern binding fallthrough, expanding immutable exact tuples within `budget`.
///
/// The budget is shared by unions, intersections, and nested patterns so that their cumulative
/// expansion cannot exceed the configured limits.
fn try_sequence_pattern_binding_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &SequencePatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    budget: &mut PatternExpansionBudget,
) -> Result<Type<'db>, ()> {
    let resolved = subject_ty.resolve_type_alias(db);
    let narrowed = match resolved {
        Type::Union(union) => try_map_union_elements(db, env, union, |element| {
            try_sequence_pattern_binding_fallthrough_type(db, env, kind, element, budget)
        })?,
        Type::Intersection(intersection) => {
            let mut failed = false;
            let narrowed = intersection.map_positive(db, env, |element| {
                try_sequence_pattern_binding_fallthrough_type(db, env, kind, *element, budget)
                    .unwrap_or_else(|()| {
                        failed = true;
                        *element
                    })
            });
            if failed {
                return Err(());
            }
            narrowed
        }
        Type::TypeVar(typevar)
            if typevar
                .typevar(db)
                .upper_bound(db, env)
                .is_some_and(|bound| {
                    pattern_fallthrough_type(
                        db,
                        env,
                        &PatternPredicateKind::Sequence(kind.clone()),
                        bound,
                    )
                    .is_never()
                }) =>
        {
            Type::Never
        }
        _ if resolved.exact_tuple_instance_spec(db).is_some() => {
            exact_tuple_sequence_pattern_fallthrough_type(db, env, kind, resolved, budget)?
                .unwrap_or_else(|| {
                    pattern_fallthrough_type(
                        db,
                        env,
                        &PatternPredicateKind::Sequence(kind.clone()),
                        resolved,
                    )
                })
        }
        // An irrefutable sequence pattern can only fail if the subject is not eligible for sequence
        // matching. Unlike length and indexed-element facts, eligibility is unaffected by mutation.
        _ if kind.is_irrefutable() => IntersectionBuilder::new(db, env)
            .add_positive(resolved)
            .add_negative(sequence_pattern_type_builder(db, env).build())
            .build(),
        _ => resolved,
    };

    if narrowed == resolved {
        Ok(subject_ty)
    } else {
        Ok(narrowed)
    }
}

/// Apply class-pattern binding fallthrough, narrowing the attributes extracted by subpatterns.
///
/// When every subject value is an instance of the pattern class, the pattern can only fail
/// because one of its subpatterns fails. Like exact tuples, the remaining type is a union with one
/// alternative per subpattern, recording what remains of that subpattern's attribute in a
/// synthesized protocol.
///
/// In the example below, the subject type remaining after the first case is
/// `Box & <Protocol with members 'value'>`, where `value` is `str`, so the second case is
/// exhaustive.
///
/// ```python
/// @dataclass
/// class Box:
///     value: int | str
///
/// def f(box: Box) -> str:
///     match box:
///         case Box(value=int()):
///             return "int"
///         case Box(value=str()):
///             return "str"
/// ```
///
/// Unlike the exhaustiveness check for a single case, this relies on the attribute keeping its
/// value from one case to the next, so it only narrows attributes accepted by
/// [`stable_member_type`].
///
/// Negative narrowing of the subject in later case bodies still subtracts only the definite-match
/// type of a class pattern. Using this fallthrough there would make the narrowing consistent, but
/// the protocol intersections would also appear in the displayed type of every subject matched by
/// several class patterns.
fn try_class_pattern_binding_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    pattern: &PatternPredicateKind<'db>,
    kind: &ClassPatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    budget: &mut PatternExpansionBudget,
) -> Result<Type<'db>, ()> {
    // Without subpatterns or a known class, there are no attributes to narrow, so there is no
    // need to split a union subject into its elements.
    let class = match class_pattern_coverage_target(db, kind) {
        Type::ClassLiteral(class) if !kind.is_empty() => class,
        _ => return Ok(pattern_fallthrough_type(db, env, pattern, subject_ty)),
    };

    let resolved = subject_ty.resolve_type_alias(db);
    let narrowed = match resolved {
        Type::Union(union) => try_map_union_elements(db, env, union, |element| {
            try_class_pattern_binding_fallthrough_type(db, env, pattern, kind, element, budget)
        })?,
        // A type variable bounded by a union, such as `T: Box | int`, is not a subtype of the
        // pattern class, but each of its values belongs to an element of the bound.
        Type::TypeVar(typevar)
            if let Some(bound @ Type::Union(_)) = typevar.typevar(db).upper_bound(db, env) =>
        {
            let remaining =
                try_class_pattern_binding_fallthrough_type(db, env, pattern, kind, bound, budget)?;
            if remaining == bound {
                resolved
            } else {
                IntersectionBuilder::new(db, env)
                    .add_positive(resolved)
                    .add_positive(remaining)
                    .build()
            }
        }
        _ => class_instance_pattern_fallthrough_type(
            db, env, pattern, kind, class, resolved, budget,
        )?,
    };

    Ok(if narrowed == resolved {
        subject_ty
    } else {
        narrowed
    })
}

/// Return the part of `subject_ty`, which is not a union, that can remain after a class pattern
/// for `class` fails.
fn class_instance_pattern_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    pattern: &PatternPredicateKind<'db>,
    kind: &ClassPatternPredicateKind<'db>,
    class: ClassLiteral<'db>,
    subject_ty: Type<'db>,
    budget: &mut PatternExpansionBudget,
) -> Result<Type<'db>, ()> {
    let fallthrough = || pattern_fallthrough_type(db, env, pattern, subject_ty);

    let class_instance_ty = Type::instance(db, env, class.top_materialization(db));
    if !subject_ty.is_subtype_of(db, env, class_instance_ty) {
        return Ok(fallthrough());
    }

    let positional_sources =
        class_pattern_positional_sources(db, env, class, kind.positional.len());
    let subpatterns =
        kind.positional
            .iter()
            .zip(positional_sources)
            .chain(kind.keywords.iter().map(|keyword| {
                (
                    &keyword.pattern,
                    ClassPatternPositionalSource::Attribute(keyword.attr.clone()),
                )
            }));

    let mut alternatives = Vec::new();
    for (subpattern, source) in subpatterns {
        let alternative = match source {
            ClassPatternPositionalSource::MatchSelf => {
                let remaining =
                    try_pattern_binding_fallthrough_type(db, env, subpattern, subject_ty, budget)?;
                if remaining == subject_ty {
                    return Ok(fallthrough());
                }
                remaining
            }
            ClassPatternPositionalSource::Attribute(name) => {
                let Some(member_ty) = stable_member_type(db, env, subject_ty, name.as_str()) else {
                    return Ok(fallthrough());
                };
                // As with gradual tuple elements, a dynamic attribute type would only carry
                // `Any & ~int`-style remainders into later cases.
                if any_over_type(db, env, member_ty, true, |ty| ty.is_dynamic()) {
                    return Ok(fallthrough());
                }
                let remaining =
                    try_pattern_binding_fallthrough_type(db, env, subpattern, member_ty, budget)?;
                // If this subpattern can fail for any attribute value, the failure of the class
                // pattern tells us nothing about the subject.
                if remaining == member_ty {
                    return Ok(fallthrough());
                }
                if remaining.is_never() {
                    continue;
                }
                IntersectionBuilder::new(db, env)
                    .add_positive(subject_ty)
                    .add_positive(Type::protocol_with_readonly_members(
                        db,
                        env,
                        [(name.as_str(), remaining)],
                    ))
                    .build()
            }
            ClassPatternPositionalSource::Unknown => return Ok(fallthrough()),
        };
        if alternative.is_never() {
            continue;
        }
        // Exceeding the limit only loses precision for this class pattern, so fall back locally
        // instead of discarding tuple expansions elsewhere in the traversal.
        if !budget.add_class_alternative() {
            return Ok(fallthrough());
        }
        alternatives.push(alternative);
    }

    Ok(UnionType::from_elements(db, env, alternatives))
}

/// Map every element of `union` within the expansion budget, then rebuild the union.
///
/// Wait until every element fits within the expansion budget before simplifying the union.
/// Otherwise, we can spend substantial time simplifying a partial result that must be discarded
/// when a later element exceeds the budget.
fn try_map_union_elements<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    union: UnionType<'db>,
    mut map: impl FnMut(Type<'db>) -> Result<Type<'db>, ()>,
) -> Result<Type<'db>, ()> {
    let mapped = union
        .elements(db)
        .iter()
        .map(|element| map(*element))
        .collect::<Result<Vec<_>, ()>>()?;
    let mut mapped = mapped.into_iter();
    union.try_map(db, env, |_| mapped.next()).ok_or(())
}

const MAX_EXACT_TUPLE_PATTERN_ALTERNATIVES: usize = 64;
const MAX_EXACT_TUPLE_PATTERN_ELEMENTS: usize = 4_096;
const MAX_CLASS_PATTERN_ALTERNATIVES: usize = 64;

/// Limits the cumulative alternatives created by one pattern traversal.
///
/// Exact tuples and class patterns are counted separately, so that class patterns cannot use up
/// the tuple limit.
#[derive(Default)]
struct PatternExpansionBudget {
    alternatives: usize,
    elements: usize,
    class_alternatives: usize,
}

impl PatternExpansionBudget {
    fn add_class_alternative(&mut self) -> bool {
        self.class_alternatives += 1;
        self.class_alternatives <= MAX_CLASS_PATTERN_ALTERNATIVES
    }

    fn add_alternative(&mut self, elements: usize) -> Result<(), ()> {
        self.alternatives += 1;
        self.elements = self.elements.saturating_add(elements);
        if self.alternatives > MAX_EXACT_TUPLE_PATTERN_ALTERNATIVES
            || self.elements > MAX_EXACT_TUPLE_PATTERN_ELEMENTS
        {
            Err(())
        } else {
            Ok(())
        }
    }
}

/// Return the part of an exact fixed-length tuple that can remain after a sequence pattern fails.
///
/// A pattern fails if any aligned element pattern fails. Represent that as a union with one tuple
/// alternative per element. Large expansions and gradual tuples keep the synthesized-protocol
/// representation used by the general fallthrough path.
fn exact_tuple_sequence_pattern_fallthrough_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &SequencePatternPredicateKind<'db>,
    subject_ty: Type<'db>,
    budget: &mut PatternExpansionBudget,
) -> Result<Option<Type<'db>>, ()> {
    if kind.split_around_star().is_some() {
        return Ok(None);
    }

    let Some(tuple) = subject_ty.exact_tuple_instance_spec(db) else {
        return Ok(None);
    };
    let Some(tuple) = tuple.as_fixed_length() else {
        return Ok(None);
    };
    if tuple
        .all_elements()
        .iter()
        .any(|element| any_over_type(db, env, *element, true, |ty| ty.is_dynamic()))
    {
        return Ok(None);
    }
    if tuple.len() != kind.patterns.len() {
        return Ok(Some(subject_ty));
    }
    let mut alternatives = Vec::new();
    for (index, (element, pattern)) in tuple
        .iter_all_elements()
        .zip(kind.patterns.iter())
        .enumerate()
    {
        let remaining = try_pattern_binding_fallthrough_type(db, env, pattern, element, budget)?;
        if remaining == element {
            return Ok(Some(subject_ty));
        }
        if remaining.is_never() {
            continue;
        }

        budget.add_alternative(tuple.len())?;
        let mut elements = tuple.all_elements().to_vec();
        elements[index] = remaining;
        alternatives.push(Type::heterogeneous_tuple(db, env, elements));
    }

    Ok(Some(UnionType::from_elements(db, env, alternatives)))
}

/// Return whether every possible value of `ty` belongs to the same enum as `right`, including
/// bounded type variables nested inside unions or intersections.
fn is_same_enum_pattern_domain<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    ty: Type<'db>,
    right: EnumLiteralType<'db>,
) -> bool {
    if is_same_enum_domain(db, env, ty, right) {
        return true;
    }

    match ty.resolve_type_alias(db) {
        Type::TypeVar(typevar) => typevar
            .typevar(db)
            .upper_bound(db, env)
            .is_some_and(|bound| is_same_enum_domain(db, env, bound, right)),
        Type::Union(union) => union
            .elements(db)
            .iter()
            .all(|element| is_same_enum_pattern_domain(db, env, *element, right)),
        Type::Intersection(intersection) => intersection
            .positive(db)
            .iter()
            .any(|element| is_same_enum_pattern_domain(db, env, *element, right)),
        _ => false,
    }
}

/// Return the definite-match type when it does not depend on the current subject type.
///
/// `None` means that callers must use subject-aware analysis instead of falling back to the
/// context-free approximation. In particular, attribute class patterns can depend on members of
/// the static subject type.
fn subject_independent_definite_match_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
) -> Option<Type<'db>> {
    match kind {
        PatternPredicateKind::Class(kind) => {
            let class_ty = class_pattern_coverage_target(db, kind);
            match class_ty {
                Type::ClassLiteral(class) if kind.is_empty() => {
                    let class_instance_ty =
                        Type::instance(db, env, class.unknown_specialization(db))
                            .top_materialization(db, env);
                    let typed_dict_adds_runtime_matches =
                        typed_dict_matches_class_pattern(db, env, class)
                            && !Type::object().is_subtype_of(db, env, class_instance_ty);
                    (!typed_dict_adds_runtime_matches).then_some(class_instance_ty)
                }
                Type::ClassLiteral(_) => None,
                Type::SpecialForm(SpecialFormType::CollectionsAbcCallable) if kind.is_empty() => {
                    Some(Type::Callable(CallableType::top(db)))
                }
                _ => Some(Type::Never),
            }
        }
        PatternPredicateKind::Sequence(kind) => {
            build_definite_sequence_pattern_type(db, env, kind, |pattern| {
                subject_independent_definite_match_pattern_type(db, env, pattern)
            })
        }
        PatternPredicateKind::Mapping(kind) => {
            if kind.is_irrefutable() {
                Some(mapping_pattern_type(db, env))
            } else {
                None
            }
        }
        PatternPredicateKind::Or(patterns) => patterns
            .iter()
            .map(|pattern| subject_independent_definite_match_pattern_type(db, env, pattern))
            .collect::<Option<Vec<_>>>()
            .map(|types| UnionType::from_elements(db, env, types)),
        PatternPredicateKind::As(Some(pattern), _) => {
            subject_independent_definite_match_pattern_type(db, env, pattern)
        }
        PatternPredicateKind::Value(_) => None,
        _ => Some(definite_match_pattern_type(db, env, kind)),
    }
}

/// Return the values that are guaranteed to match `kind`.
///
/// Reachability and negative narrowing can only subtract this under-approximation.
pub(crate) fn definite_match_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &PatternPredicateKind<'db>,
) -> Type<'db> {
    match kind {
        PatternPredicateKind::Singleton(singleton) => singleton_pattern_type(db, env, *singleton),
        PatternPredicateKind::Value(value) => {
            let ty = infer_same_file_expression_type(db, *value, TypeContext::default());
            // Only return the type if it's guaranteed to match itself.
            // Otherwise, we can't definitively exclude it from subsequent patterns.
            let policy = ComparisonSoundnessPolicy::from_analysis_settings(
                db.analysis_settings(value.file(db)),
            );
            if equality_truthiness(db, env, ty, ty, policy) == Truthiness::AlwaysTrue {
                ty
            } else {
                Type::Never
            }
        }
        PatternPredicateKind::Class(kind) => {
            let class_ty = class_pattern_coverage_target(db, kind);
            match class_ty {
                Type::ClassLiteral(class) if kind.is_empty() => {
                    Type::instance(db, env, class.unknown_specialization(db))
                        .top_materialization(db, env)
                }
                Type::SpecialForm(SpecialFormType::CollectionsAbcCallable) if kind.is_empty() => {
                    Type::Callable(CallableType::top(db))
                }
                _ => Type::Never,
            }
        }
        PatternPredicateKind::Mapping(kind) => {
            if kind.is_irrefutable() {
                mapping_pattern_type(db, env)
            } else {
                Type::Never
            }
        }
        PatternPredicateKind::Sequence(kind) => definite_sequence_pattern_type(db, env, kind),
        PatternPredicateKind::Or(predicates) => UnionType::from_elements(
            db,
            env,
            predicates
                .iter()
                .map(|p| definite_match_pattern_type(db, env, p)),
        ),
        PatternPredicateKind::As(pattern, _) => pattern
            .as_deref()
            .map(|p| definite_match_pattern_type(db, env, p))
            .unwrap_or_else(Type::object),
        PatternPredicateKind::Star(_) => Type::object(),
    }
}

/// Return the values that are guaranteed to match a sequence pattern.
fn definite_sequence_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &SequencePatternPredicateKind<'db>,
) -> Type<'db> {
    build_definite_sequence_pattern_type(db, env, kind, |pattern| {
        Some(definite_match_pattern_type(db, env, pattern))
    })
    .unwrap_or(Type::Never)
}

fn build_definite_sequence_pattern_type<'db>(
    db: &'db dyn Db,
    env: &ProgramEnvironment<'db>,
    kind: &SequencePatternPredicateKind<'db>,
    mut element_type: impl FnMut(&PatternPredicateKind<'db>) -> Option<Type<'db>>,
) -> Option<Type<'db>> {
    if kind.is_irrefutable() {
        return Some(sequence_pattern_type_builder(db, env).build());
    }

    if let Some((prefix, suffix)) = kind.split_around_star() {
        let prefix_types = prefix
            .iter()
            .map(&mut element_type)
            .collect::<Option<Vec<_>>>()?;
        let suffix_types = suffix
            .iter()
            .map(&mut element_type)
            .collect::<Option<Vec<_>>>()?;
        return Some(Type::tuple(TupleType::mixed(
            db,
            env,
            prefix_types,
            Type::object(),
            suffix_types,
        )));
    }

    let element_types: Vec<_> = kind
        .patterns
        .iter()
        .map(element_type)
        .collect::<Option<_>>()?;

    if element_types.iter().any(Type::is_never) {
        Some(Type::Never)
    } else {
        Some(exact_sequence_pattern_type(
            db,
            env,
            element_types.into_iter(),
        ))
    }
}
