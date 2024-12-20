DEFAULT_SERVER_PORT = 5005

DEFAULT_SERVER_INTERFACE = "0.0.0.0"

DEFAULT_SERVER_FORMAT = "{}://localhost:{}"

DEFAULT_SERVER_URL = DEFAULT_SERVER_FORMAT.format("http", DEFAULT_SERVER_PORT)

DEFAULT_INTERACTIVE_SERVER_URL = "{}://localhost:{}"

DEFAULT_NLU_FALLBACK_THRESHOLD = 0.3

DEFAULT_NLU_FALLBACK_AMBIGUITY_THRESHOLD = 0.1

DEFAULT_CORE_FALLBACK_THRESHOLD = 0.3

DEFAULT_MAX_HISTORY = None  # Core policy history is unbounded by default.

DEFAULT_RESPONSE_TIMEOUT = 60 * 60  # 1 hour

DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes

DEFAULT_STREAM_READING_TIMEOUT = 10  # in seconds

DEFAULT_LOCK_LIFETIME = 60  # in seconds

DEFAULT_KEEP_ALIVE_TIMEOUT = 120  # in seconds

BEARER_TOKEN_PREFIX = "Bearer "

# The lowest priority is intended to be used by machine learning policies.
DEFAULT_POLICY_PRIORITY = 1

# The priority of intent-prediction policies.
# This should be below all rule based policies but higher than ML
# based policies. This enables a loop inside ensemble where if none
# of the rule based policies predict an action and intent prediction
# policy predicts one, its prediction is chosen by the ensemble and
# then the ML based policies are again run to get the prediction for
# an actual action. To prevent an infinite loop, intent prediction
# policies only predict an action if the last event in
# the tracker is of type `UserUttered`. Hence, they make at most
# one action prediction in each conversation turn. This allows other
# policies to predict a winning action prediction.
UNLIKELY_INTENT_POLICY_PRIORITY = DEFAULT_POLICY_PRIORITY + 1

# The priority intended to be used by memoization policies.
# It is higher than default to prioritize training stories.
MEMOIZATION_POLICY_PRIORITY = UNLIKELY_INTENT_POLICY_PRIORITY + 1
# The priority of the `RulePolicy` is higher than all other policies since
# rule execution takes precedence over training stories or predicted actions.
RULE_POLICY_PRIORITY = MEMOIZATION_POLICY_PRIORITY + 1

DIALOGUE = "dialogue"

# RabbitMQ message property header added to events published using `rasa export`
RASA_EXPORT_PROCESS_ID_HEADER_NAME = "rasa-export-process-id"

# Name of the environment variable defining the PostgreSQL schema to access. See
# https://www.postgresql.org/docs/9.1/ddl-schemas.html for more details.
POSTGRESQL_SCHEMA = "POSTGRESQL_SCHEMA"

# Names of the environment variables defining PostgreSQL pool size and max overflow
POSTGRESQL_POOL_SIZE = "SQL_POOL_SIZE"
POSTGRESQL_MAX_OVERFLOW = "SQL_MAX_OVERFLOW"

# File names for testing
CONFUSION_MATRIX_STORIES_FILE = "story_confusion_matrix.png"
REPORT_STORIES_FILE = "story_report.json"
FAILED_STORIES_FILE = "failed_test_stories.yml"
SUCCESSFUL_STORIES_FILE = "successful_test_stories.yml"
STORIES_WITH_WARNINGS_FILE = "stories_with_warnings.yml"

POLICY_PRIORITY = "priority"
POLICY_FEATURIZER = "featurizer"
POLICY_MAX_HISTORY = "max_history"

DEFAULT_PROTOCOL = "UDP"
DEFAULT_SYSLOG_HOST = "localhost"
DEFAULT_SYSLOG_PORT = 514

COMPRESS_ACTION_SERVER_REQUEST_ENV_NAME = "COMPRESS_ACTION_SERVER_REQUEST"
DEFAULT_COMPRESS_ACTION_SERVER_REQUEST = False
