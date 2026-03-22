import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# ===============================
# DATABASE CONFIG
# ===============================

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL=os.getenv("DEEPSEEK_MODEL")
DEEPSEEK_BASE_URL=os.getenv("DEEPSEEK_BASE_URL")
# ===============================
# LLM CONFIG
# ===============================

OPENAI_API_KEY_GPT = os.getenv("OPENAI_API_KEY_GPT")

LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
# ============================================================================
# PIPELINE RUN CONTROL
# ============================================================================
RUN_LLM_CLASSIFICATION = False  # Set to False to skip Step 5 (run merging only)
# ===============================
# GENERAL CONFIG
# ===============================

SCHEMA = "PROCESSING"
FUZZY_SIMILARITY_THRESHOLD = 85
DEFAULT_MIN_TIMESTAMP = "2002-01-01 00:00:00"
LLM_BATCH_SIZE = 20

# ===============================
# CATEGORY CONFIG
# ===============================

CATEGORY_CONFIG = {
    "TOOLS": {
        "merged_table": "NON_STANDARDIZED_DATA_TOOLS",
        "standardization_table": "TOOL_STANDARDIZATION_RESULT",
        "masterdata_table": "MASTERDATA_TOOLS",
        "high_confidence_table": "EXTRACTED_TOOLS_WITH_HIGH_CONFIDENCE",
        "item_column": "TOOL",
        "extracted_column": "EXTRACTED_TOOL_NAME",
        "data_json_column": "TOOLS_DATA",
        "json_item_key": "tool",
        "min_confidence": 0.7
    },
    "TECHNOLOGIES": {
        "merged_table": "NON_STANDARDIZED_DATA_TECHNOLOGIES",
        "standardization_table": "TECHNOLOGY_STANDARDIZATION_RESULT",
        "masterdata_table": "MASTERDATA_TECHNOLOGY",
        "high_confidence_table": "EXTRACTED_TECHNOLOGIES_WITH_HIGH_CONFIDENCE",
        "item_column": "TECHNOLOGY",
        "extracted_column": "EXTRACTED_TECHNOLOGY_NAME",
        "data_json_column": "TECHNOLOGIES_DATA",
        "json_item_key": "technology",
        "min_confidence": 0.6
    },
    "COMPLIANCES": {
        "merged_table": "NON_STANDARDIZED_DATA_COMPLIANCES",
        "standardization_table": "COMPLIANCE_STANDARDIZATION_RESULT",
        "masterdata_table": "MASTERDATA_COMPLIANCE",
        "high_confidence_table": "EXTRACTED_COMPLIANCES_WITH_HIGH_CONFIDENCE",
        "item_column": "COMPLIANCE",
        "extracted_column": "EXTRACTED_COMPLIANCE_NAME",
        "data_json_column": "COMPLIANCES_DATA",
        "json_item_key": "compliance",
        "min_confidence": 0.7
    }
}


# ===============================
# EXTRA PIPELINE TABLE CONFIG
# ===============================

PIPELINE_TABLES = {

    # New LLM classification output
    "LLM_CLASSIFICATION": "MASTERDATA_LLM_CLASSIFICATION_VALIDATION",

    # Processed final output tables
    "PROCESSED": {
        "TOOLS": "PROCESSED_TOOLS_DATA",
        "TECHNOLOGIES": "PROCESSED_TECHNOLOGY_DATA",
        "COMPLIANCES": "PROCESSED_COMPLIANCE_DATA"
    },
    # (Optional) labeled data source if you want later
    "LABELED": "PROCESSED_LABELED_DATA"
}


TOOLS_ATTRIBUTE_TABLES = {
    "SOURCE_CLASSIFICATION": "MASTERDATA_LLM_CLASSIFICATION_VALIDATION",
    "INTERMEDIATE": "MASTERDATA_TOOLS_ATTRIBUTES_INTERMEDIATE",
    "VALIDATION": "MASTERDATA_TOOLS_ATTRIBUTES_VALIDATION"
}

TECHNOLOGY_ATTRIBUTE_TABLES = {
    "SOURCE_CLASSIFICATION": "MASTERDATA_LLM_CLASSIFICATION_VALIDATION",
    "INTERMEDIATE": "MASTERDATA_TECHNOLOGY_ATTRIBUTES_INTERMEDIATE",
    "VALIDATION": "MASTERDATA_TECHNOLOGY_ATTRIBUTES_VALIDATION"
}

# ===============================
# COMPLIANCE ATTRIBUTE PIPELINE CONFIG
# ===============================

COMPLIANCE_ATTRIBUTE_TABLES = {
    "SOURCE_CLASSIFICATION": "MASTERDATA_LLM_CLASSIFICATION_VALIDATION",
    "INTERMEDIATE": "MASTERDATA_COMPLIANCE_ATTRIBUTES_INTERMEDIATE",
    "VALIDATION": "MASTERDATA_COMPLIANCE_ATTRIBUTES_VALIDATION"
}