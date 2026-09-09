from enum import Enum


class Intent(str, Enum):
    BILLING_DISPUTE = "billing_dispute"
    PLAYBACK_BUG = "playback_bug"
    ACCOUNT_ACCESS = "account_access"
    CANCELLATION_RETENTION = "cancellation_retention"
    FEATURE_QUESTION = "feature_question"
    OTHER = "other"
