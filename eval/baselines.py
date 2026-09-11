import re

from domain.intents import Intent
from domain.models import ClassificationResult, DraftReply, Message

_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    Intent.CANCELLATION_RETENTION: (
        "cancel",
        "unsubscribe",
        "downgrade",
        "close my account",
        "delete my account",
    ),
    Intent.BILLING_DISPUTE: (
        "charge",
        "charged",
        "bill",
        "billing",
        "refund",
        "invoice",
        "overcharged",
        "payment",
    ),
    Intent.ACCOUNT_ACCESS: (
        "password",
        "log in",
        "login",
        "locked out",
        "sign in",
        "hacked",
        "two-factor",
        "reset my account",
    ),
    Intent.PLAYBACK_BUG: (
        "skip",
        "buffer",
        "crash",
        "freeze",
        "frozen",
        "won't play",
        "wont play",
        "stuck",
        "lag",
        "glitch",
        "stopped working",
    ),
    Intent.FEATURE_QUESTION: (
        "how do i",
        "how to",
        "does spotify have",
        "is there a way",
        "feature request",
        "can you add",
    ),
}

_CANNED_REPLIES: dict[Intent, str] = {
    Intent.BILLING_DISPUTE: (
        "Thanks for reaching out about your billing. We're looking into the charge on your "
        "account and will follow up with details shortly."
    ),
    Intent.PLAYBACK_BUG: (
        "Sorry for the trouble with playback. Try updating the app and clearing its cache; "
        "let us know if the issue continues."
    ),
    Intent.ACCOUNT_ACCESS: (
        "We can help you regain access. Please use the 'Forgot password' link on the login "
        "page to reset your credentials."
    ),
    Intent.CANCELLATION_RETENTION: (
        "We're sorry to see you consider leaving. You can manage or cancel your subscription "
        "anytime from your account settings."
    ),
    Intent.FEATURE_QUESTION: (
        "Thanks for the question! Check out our Help Center for details on this feature, or "
        "let us know if you need more info."
    ),
    Intent.OTHER: "Thanks for reaching out. A member of our team will review your message and get back to you.",
}


def _matches_keyword(keyword: str, lowered_text: str) -> bool:
    # Word-boundary match, not plain substring: "charge" must not match inside "charger",
    # "bill" must not match inside "billboard" or the name "Bill".
    return re.search(rf"\b{re.escape(keyword)}\b", lowered_text) is not None


class KeywordClassifier:
    def classify(self, message: Message) -> ClassificationResult:
        lowered = message.text.lower()
        for intent, keywords in _KEYWORDS.items():
            for keyword in keywords:
                if _matches_keyword(keyword, lowered):
                    return ClassificationResult(
                        intent=intent, confidence=1.0, rationale=f"matched keyword '{keyword}'"
                    )
        return ClassificationResult(intent=Intent.OTHER, confidence=1.0, rationale="no keyword matched")


class CannedReplyDrafter:
    def draft(self, message: Message, intent: Intent) -> DraftReply:
        return DraftReply(text=_CANNED_REPLIES[intent], grounded_on=[])
