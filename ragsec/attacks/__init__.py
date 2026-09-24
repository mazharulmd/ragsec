"""Knowledge-base poisoning attacks used to benchmark defenses."""

from ragsec.attacks.adaptive import AdaptivePoisonAttack
from ragsec.attacks.base import Attack, AttackTarget, content_keywords, strip_question
from ragsec.attacks.flooding import ContextFloodingAttack
from ragsec.attacks.injection import PromptInjectionAttack
from ragsec.attacks.poisoned_rag import PoisonedRAGAttack
from ragsec.attacks.trigger_backdoor import TriggerBackdoorAttack

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    PoisonedRAGAttack.name: PoisonedRAGAttack,
    AdaptivePoisonAttack.name: AdaptivePoisonAttack,
    TriggerBackdoorAttack.name: TriggerBackdoorAttack,
    PromptInjectionAttack.name: PromptInjectionAttack,
    ContextFloodingAttack.name: ContextFloodingAttack,
}


def get_attack(name: str, **kwargs: object) -> Attack:
    try:
        cls = ATTACK_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown attack {name!r}; choose from {sorted(ATTACK_REGISTRY)}") from exc
    return cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "ATTACK_REGISTRY",
    "AdaptivePoisonAttack",
    "Attack",
    "AttackTarget",
    "ContextFloodingAttack",
    "PoisonedRAGAttack",
    "PromptInjectionAttack",
    "TriggerBackdoorAttack",
    "content_keywords",
    "get_attack",
    "strip_question",
]
