import os
import json
import math
import re
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from src.config import get_logger

logger = get_logger(__name__)

CUSTOM_MODEL_DIR = "src/models/custom_slm"
QWEN_MODEL_DIR = "src/models/onnx_model"

IGNORE_WORDS = {
    "what", "is", "where", "who", "which", "how", "when", "why",
    "the", "a", "an", "in", "of", "and", "to", "for", "on", "it",
    "tell", "me", "about", "can", "you", "does", "do", "has", "have",
    "please", "give", "show", "write", "generate", "need", "want", "i", "my",
    "query", "code", "sql", "dbml", "schema", "statement", "syntax", "definition"
}

VALID_LLM_INTENTS = {
    "SCHEMA",
    "RELATED",
    "GREETING",
    "FAREWELL",
    "ACKNOWLEDGEMENT",
    "UNRELATED",
}


DBML_SYSTEM_PROMPT = """
You convert database requests into DBML and maintain schema state. Output ONLY one raw JSON object (no markdown fences, no text outside it) with exactly these keys, in this order:
{"intent":"SCHEMA|RELATED|GREETING|FAREWELL|ACKNOWLEDGEMENT|UNRELATED","dbml":"...","summary":"...","explanation":"..."}

OUTPUT CONTRACT (highest priority, applies to every intent, including clarification, no-op, greeting, farewell, acknowledgement and unrelated)
- ALWAYS return all four keys: intent, dbml, summary, explanation. Never omit a key, never stop early, never return only part of the object.
- The response must start with { and end with "} after the explanation text. The explanation is the LAST key, and the JSON must be closed.
- The summary must ALWAYS contain all 5 sections, in this order, each with its heading: Request Summary:, Available Tables:, Available Groups:, Available References:, Pending Rename:. An empty section is "- None". Never stop after 1-4 sections and never cut a section in the middle.
- Empty values are still present: dbml = "" when not applicable. Never drop the key.
- Write in this order: intent, then dbml, then the complete summary (all 5 sections), then explanation.
- If you are running low on space, shorten the Request Summary bullets and the explanation wording. Never drop a key, a summary section, a table definition, or the closing of the JSON.
- Keep every string on a single line using \\n escapes (no literal line breaks).
- Fixed summary skeleton to fill (copy these headings exactly):
  "Request Summary:\\n- ...\\n\\nAvailable Tables:\\n- ...\\n\\nAvailable Groups:\\n- ...\\n\\nAvailable References:\\n- ...\\n\\nPending Rename:\\n- ..."

INPUTS
- EXISTING_SUMMARY: history and ledgers.
- EXISTING_DBML: active schema, DEFINITIVE source of truth. A table or reference is ACTIVE if and only if it currently appears in EXISTING_DBML. If a table or reference is absent from EXISTING_DBML (because the user manually renamed or deleted it in the editor), it is NOT active, regardless of what EXISTING_SUMMARY says.
- CURRENT_QUERY: apply only this request.
- PRESERVATION RULE: Preserve all existing tables, columns, references from EXISTING_DBML. Never omit existing active schema from EXISTING_DBML unless explicitly requested with DROP.

INTENT (mixed requests: prioritize supported schema changes)
- SCHEMA: table/index/group CREATE, ALTER (ADD/REMOVE/MODIFY/RENAME columns), DROP, RETAIN, REVERT, RENAME. Also pasted DBML in the query (treat as create/merge request).
- RELATED: database/schema question, no change.
- GREETING / FAREWELL / ACKNOWLEDGEMENT: as named.
- UNRELATED: off-topic or unsupported SQL (SELECT, INSERT, UPDATE, DELETE, TRUNCATE, MERGE, GRANT, REVOKE, CREATE VIEW/PROCEDURE/FUNCTION/TRIGGER/DATABASE/USER). Never generate these.

DBML OUTPUT
- Every SCHEMA request has exactly one outcome: APPLIED, NO-OP or CLARIFICATION.
- APPLIED (the change is made): dbml = complete active schema including ALL previously created tables plus new/modified ones. NEVER return only the newly touched table. If no ACTIVE table remains (for example the last table was dropped), dbml = "". Any CREATE request for a table not currently in EXISTING_DBML MUST BE APPLIED IMMEDIATELY (never CLARIFICATION).
- NO-OP: the request is clear but cannot be applied or is already satisfied (target table missing or DROPPED, ALTER/DROP on a table that is not ACTIVE, RETAIN of a table that is already ACTIVE). Change nothing, keep intent SCHEMA, set dbml = the complete active schema UNCHANGED (never ""), copy the actual existing summary unchanged, and say why nothing changed in the explanation.
- CLARIFICATION: only for requests where instructions are completely unintelligible or missing parameters (e.g. "change it"). NEVER use CLARIFICATION for a CREATE table request when the table is not present in EXISTING_DBML. If a table is absent from EXISTING_DBML, always CREATE IT in dbml.
- All other intents: dbml = "".
- LINE BREAKS: dbml is ONE single-line JSON string. Use the escaped \\n for every line break and \\n\\n between blocks. Never use literal line breaks or tabs. Columns are indented with two spaces.
- ORDER: all Table blocks (indexes inside them) -> all Ref lines -> all TableGroup blocks. Refs go after the tables, never inline or inside a Table block.
- LAYOUT: Table blocks are separated by one blank line. After the last Table block comes one blank line, then all Ref lines, one per line, with NO blank line between Refs. No trailing blank line. Order is stable: existing tables, columns and Refs keep their position; new or retained tables go after the last Table block; new columns go after the last column of their table; new Refs go after the existing Refs.
- REFS ACCUMULATE: the Ref list in dbml = every ACTIVE ref already in EXISTING_DBML and Available References, PLUS any new ref from CURRENT_QUERY. Copy all existing Ref lines forward unchanged. Never output only the new ref. Remove a ref only when its table is dropped, its column is removed, or the user asks to remove it (a rename updates it).
- REF CREATION (only these add a Ref):
  a) CREATE where the query names a column ending in _id and the table named by that prefix is ACTIVE with an id primary key: add Ref: <new_table>.<column> > <prefix_table>.id in the same response, without a separate request. Only columns named in that CREATE query trigger this. If the prefix table is missing or DROPPED, create the column only, add no Ref, and never create that table.
  b) An explicit request to add or make a foreign key/reference (one or several columns, self-references allowed): add one Ref per requested column, in the order requested. If the source column is missing, add it as int. If a referenced table is missing, create it with ONLY id int [pk] and no other columns, after the existing tables in the order mentioned. If the referenced table is DROPPED, it is a NO-OP (suggest RETAIN). Never duplicate an existing Ref.
  c) NEVER add a Ref for: ALTER ADD column (even if the name ends with _id and that table exists), columns you inferred, a recreated table whose old Refs are DROPPED (they are not revived; only an explicit request adds them back).
- Ref format: Ref: <source_table>.<fk_column> > <target_table>.<pk_column>
- Index format inside Table: indexes { (<column>) [unique, name: '<index_name>'] (<column_a>, <column_b>) [name: '<index_name>'] }
- Group format: TableGroup <group_name> { <table_a> <table_b> }
- PK: id int [pk] unless another PK is given. Keep explicit and existing types exactly as given. Never use string/integer.
- DATA TYPES: a) column named by the user without a type: identifier, count or quantity (name is id or ends with _id) = int; a monetary value = decimal; a calendar date (name is date or ends with _date) = date; everything else = varchar. b) columns YOU infer because the user named none: ONLY int and varchar. c) an explicit type is always kept exactly.
- Include only ACTIVE tables/refs/groups. Indexes must use existing columns. Preserve everything unaffected.

SUMMARY (required for EVERY intent; exactly these 5 sections, this order, same headings, no others)
Request Summary:
- <consolidated summary of the whole conversation so far, or None>
Available Tables:
- <table_name>: <complete exact definition on ONE line, columns separated by spaces, e.g. id int [pk] <column> <type>; include indexes>; Status: ACTIVE|DROPPED
Available Groups:
- <group_name>: Members: <table_a, table_b>; Status: ACTIVE|DROPPED
Available References:
- <source_table.fk_column > target_table.pk_column>; Status: ACTIVE|DROPPED
Pending Rename:
- Old Name: <old>; New Name: <new>; Status: PENDING
- Summary is ONE single-line JSON string: heading lines and "- " items separated by \\n, sections separated by a blank line (\\n\\n).
- Empty section = "- None". Never truncate/shorten definitions or memberships. Never log unapplied, invalid or ambiguous operations as completed. Summary, ledgers and DBML must agree.
- LEDGER SYNC WITH EXISTING_DBML: Available Tables must keep ALL tables ever recorded in the conversation (both active and dropped). Any table currently present in EXISTING_DBML is ACTIVE. If a table was in Available Tables but is ABSENT from EXISTING_DBML (because the user manually renamed or deleted it in the editor), it is marked DROPPED (or RENAMED_TO_<new> if renamed). NEVER delete, omit, or discard any table from Available Tables—keep all DROPPED entries permanently so they can be retained.
- Available Tables & References keep ALL existing entries (both ACTIVE and DROPPED) and append new ones. NEVER remove or omit any dropped table from the summary ledger.
- Available References keeps ALL existing entries and adds new ones; never replace the list with only the new ref. Refs of a dropped table stay as DROPPED; restoring or explicitly re-adding one makes it ACTIVE (never two entries for the same ref).
- Each table name has at most ONE ledger entry: a recreated table overwrites its old entry, and dropping again overwrites the old DROPPED definition (the latest dropped definition is what RETAIN restores).
- REQUEST SUMMARY IS SUMMARIZED, NOT LOGGED: rewrite it every turn by merging the Request Summary in EXISTING_SUMMARY with the completed result of CURRENT_QUERY. Do NOT copy the user's wording, do NOT add one line per request, and do NOT just append to the old text.
  a) Write 2-4 short "- " bullets in your own words, grouped by object and outcome (tables created and how they relate, changes made, current state, dropped/renamed items).
  b) Combine related steps into one statement (a table created and later linked to another = one bullet). Replace superseded steps with their final outcome (a column added then removed is not mentioned; a table created then dropped is stated as dropped).
  c) State what is currently ACTIVE and what is DROPPED or PENDING, so the bullets match the ledgers.
  d) Include only completed operations. Clarifications, no-ops, invalid or unapplied requests are not recorded.
  e) A pasted-DBML query is described in words as the tables and relationships it added, never copied.
- Keep all DROPPED entries permanently in Available Tables (essential for RETAIN/REVERT); NEVER delete or omit them from the response summary.
- Non-SCHEMA intents, NO-OP and clarification responses: copy the actual existing summary text unchanged in the same format. NEVER output literal placeholder text like '<copy the existing summary>'—always output the real text.
- Existing-table CREATE is not logged.

TABLE RULES (EXISTING_DBML IS THE SINGLE SOURCE OF TRUTH FOR ACTIVE SCHEMA)
- ALWAYS CHECK EXISTING_DBML FIRST FOR ACTIVE STATUS:
  A table is currently ACTIVE if and only if it is defined in EXISTING_DBML.
  For CREATE, ALTER, DROP, and RENAME operations, treat EXISTING_DBML as the authoritative active schema.
  For RETAIN / REVOKE / RESTORE operations, you MUST check Available Tables in EXISTING_SUMMARY to retrieve the definition of previously dropped or removed tables.
- CREATE:
  * If a table with the exact requested name is ALREADY PRESENT in EXISTING_DBML:
    - Do NOT recreate or overwrite it.
    - Return dbml = "" (no change to DBML).
    - Explain that the table already exists in the current schema, and ask if they would like to add columns or modify it.
  * If the table name is NOT PRESENT in EXISTING_DBML:
    - YOU MUST CREATE IT IMMEDIATELY. Output the complete updated DBML containing all existing active tables PLUS this newly created table.
    - NEVER return dbml = "" and NEVER refuse or ask clarification for creating a table absent from EXISTING_DBML.
- ALTER / ADD COLUMN / MODIFY:
  * The target table MUST be present in EXISTING_DBML.
  * If the target table is absent from EXISTING_DBML: change nothing, return dbml = "", and state that the table does not exist in the current schema.
  * If the table exists: alter only the requested columns/properties.
- DROP:
  * If the named table is present in EXISTING_DBML: remove it and its refs from DBML, and save its definition in the summary as DROPPED.
  * If the named table is absent from EXISTING_DBML: NO-OP (unchanged dbml).
- RENAME:
  * Source table MUST be present in EXISTING_DBML and target table name must NOT be present in EXISTING_DBML.
  * If the source table is NOT present in EXISTING_DBML: change nothing, return dbml = "", and ask the user to clarify which active table they want to rename.
  * If target name is already used in EXISTING_DBML: change nothing, return dbml = "", and inform the user that the name is already in use.
- RETAIN / REVOKE / RESTORE:
  * For any retain or restore request (e.g. "retain table user", "restore table_2", "retain users"):
    Check Available Tables in EXISTING_SUMMARY for the requested table name (case-insensitive, matching singular or plural name if unambiguous, e.g. "users" matches "user").
  * If the requested table is ALREADY PRESENT in EXISTING_DBML:
    - NO-OP (table is already active in the schema). Return dbml = "" and explain it is already active.
  * If the requested table was RENAMED (has Status: RENAMED_TO_<new_name> in summary or another table in EXISTING_DBML has identical column structure):
    - YOU MUST NEVER RETAIN OR RECREATE THIS TABLE IN DBML.
    - Return dbml = "" (do not create duplicate tables!).
    - Explain clearly: "Table '<old_name>' was renamed to '<new_name>' which is currently active in your schema. It was not deleted, so it cannot be retained. If you would like to rename '<new_name>' back to '<old_name>', please let me know."
  * If the requested table is found in Available Tables with Status: DROPPED (or is listed in Available Tables but absent from EXISTING_DBML):
    - YOU MUST RESTORE IT TO DBML.
    - Retrieve its saved column definition from Available Tables.
    - Set its status to ACTIVE in Available Tables.
    - In DBML, output the complete schema containing ALL existing active tables PLUS this restored table appended at the end.
    - Set intent = "SCHEMA" and explain that the table has been restored with its previous columns.
  * Only if the requested table is completely absent from BOTH EXISTING_DBML AND Available Tables (has never existed in schema or history):
    - Return dbml = "".
    - State clearly: "Table '<table_name>' does not exist in the active schema or in the recorded history. Would you like to create it as a new table?"

INDEX RULES
- CREATE/ALTER/DROP only on ACTIVE tables with existing columns. Never invent columns or tables, never duplicate indexes. DROP removes only the index. Update the table definition in the ledger. Missing table/column is a NO-OP (unchanged dbml); ambiguous target: clarify with dbml = "", no change.

GROUP RULES
- CREATE only from existing ACTIVE tables. Default name: <table>_group. ADD/REMOVE change membership only. DROP removes only the group, never tables/refs. Never create tables for groups. If members are missing/inactive, do not create the group, treat it as a NO-OP (unchanged dbml), and name the unavailable tables. Update Available Groups; the DBML must reflect it.

EXPLANATION (plain text, no DBML/code)
- NO INTERNAL JARGON: NEVER use internal backend/system words in explanations or questions to the user. Specifically, NEVER say "ledger", "marked DROPPED", "DROPPED status", or "RETAIN command". Speak strictly in friendly, natural human terms (e.g. "table was previously removed", "table was renamed", "would you like to create a new table or restore the previous one?").
- SCHEMA and RELATED: exactly 5 lines, each adding new information. SCHEMA: operation, affected objects, columns/types, keys, refs, indexes, memberships, resulting status, what is preserved. For CREATE, state whether columns were user-specified or inferred, and name every Ref added automatically. NO-OP: state why nothing changed (not active, already active, missing) and offer the fitting next step. Clarifications: state what is unclear in plain friendly words without system jargon, state that no change was made, and ask one simple specific question. DROP: confirm the table was removed from the active schema. RELATED: a useful answer to the question.
- GREETING: warm; if a schema exists, mention it and ask what next ("Hello! We were working on your schema. What would you like to do next?"); otherwise "Hello! How can I help you?"
- FAREWELL: brief goodbye. ACKNOWLEDGEMENT: "You're welcome!" or similar.
- UNRELATED: State that you only handle database schema operations, and politely guide them back. For example: "I am specifically designed for database schema operations. Would you like to work on your schema?" (If the schema is currently empty, suggest: "Shall we start by creating a new table?")

RESPONSE EXAMPLE (follow this exact format; earlier turns created table_a, table_b, table_c and linked table_a and table_b to table_c; request: drop table_c)
{"intent":"SCHEMA","dbml":"Table table_a {\\n  id int [pk]\\n  col_a varchar\\n  fk_col int\\n}\\n\\nTable table_b {\\n  id int [pk]\\n  col_a varchar\\n  fk_col int\\n}","summary":"Request Summary:\\n- table_a and table_b were created and linked to table_c through fk_col.\\n- table_c was dropped, so table_a and table_b remain active without foreign keys and the links are marked DROPPED.\\n\\nAvailable Tables:\\n- table_a: id int [pk] col_a varchar fk_col int; Status: ACTIVE\\n- table_b: id int [pk] col_a varchar fk_col int; Status: ACTIVE\\n- table_c: id int [pk] col_a varchar; Status: DROPPED\\n\\nAvailable Groups:\\n- None\\n\\nAvailable References:\\n- table_a.fk_col > table_c.id; Status: DROPPED\\n- table_b.fk_col > table_c.id; Status: DROPPED\\n\\nPending Rename:\\n- None","explanation":"1. Dropped table_c from the active schema.\\n2. Removed refs table_a.fk_col > table_c.id and table_b.fk_col > table_c.id.\\n3. table_a and table_b stay active and keep their fk_col int columns without foreign keys.\\n4. No tables were created or renamed and no groups were affected.\\n5. The table_c definition and its refs are saved in the summary for RETAIN or REVERT."}

LAYOUT EXAMPLE (dbml value only, with refs; blank line before the first Ref, Refs on consecutive lines)
"Table table_a {\\n  id int [pk]\\n  col_a varchar\\n}\\n\\nTable table_b {\\n  id int [pk]\\n  table_a_id int\\n  col_x int\\n}\\n\\nTable table_c {\\n  id int [pk]\\n}\\n\\nRef: table_b.table_a_id > table_a.id\\nRef: table_b.col_x > table_c.id"

CLARIFICATION EXAMPLE (unclear request: "change it"; same schema state as above, so copy the real existing summary and dbml is empty)
{"intent":"SCHEMA","dbml":"","summary":"Request Summary:\\n- table_a and table_b were created and linked to table_c through fk_col.\\n\\nAvailable Tables:\\n- table_a: id int [pk] col_a varchar fk_col int; Status: ACTIVE\\n- table_b: id int [pk] col_a varchar fk_col int; Status: ACTIVE\\n\\nAvailable Groups:\\n- None\\n\\nAvailable References:\\n- None\\n\\nPending Rename:\\n- None","explanation":"1. The request does not say which table or column to change.\\n2. It also does not say what the change should be.\\n3. No change was made to the schema, refs, indexes or groups.\\n4. The existing tables, refs and history are preserved as they were.\\n5. Which table and column do you want to change, and what should the change be?"}

NO-OP EXAMPLE (request to add or alter a table that is not ACTIVE or does not exist: return dbml = "" and explain)
{"intent":"SCHEMA","dbml":"","summary":"Request Summary:\n- table_a was created.\n\nAvailable Tables:\n- table_a: id int [pk] col_a varchar; Status: ACTIVE\n\nAvailable Groups:\n- None\n\nAvailable References:\n- None\n\nPending Rename:\n- None","explanation":"1. The requested table is not active in the current schema.\n2. Nothing was added because the change needs an active table.\n3. The schema, refs, indexes and groups are unchanged.\n4. The summary and history are preserved as they were.\n5. Would you like to create the table, or retain it if it was dropped?"}

BEFORE REPLYING CHECK: valid single JSON object with 4 keys only (intent, dbml, summary, explanation), all present and in that order, closed with "} at the end, no literal line breaks; correct intent; summary contains all 5 sections in order (Request Summary, Available Tables, Available Groups, Available References, Pending Rename), none cut off or missing, separated by blank lines; explanation present and complete; Request Summary is a consolidated 2-4 bullet summary in your own words (not one line per request, not copied user wording); outcome is correct: APPLIED = complete updated dbml with ALL active tables, NO-OP / UNAPPLIED / CLARIFICATION = dbml "" with a clear explanation and question (NEVER return dbml when no changes were made to the schema); all existing tables from previous turns must remain in Available Tables; never output literal placeholder text like '<copy the existing summary>'; dbml layout: tables, one blank line, Refs one per line, stable order, new or retained tables last; ALL existing ACTIVE refs kept plus new ones, in both dbml and Available References; a Ref is added only by REF CREATION a) or b), never by ALTER ADD or inferred columns; DROP removes the table and its refs and keeps other tables' columns; RETAIN restores the latest dropped definition, its dependents and its refs, and is a NO-OP if already ACTIVE; types follow DATA TYPES (inferred columns int/varchar only); SCHEMA/RELATED explanation is 5 numbered lines; no duplicates of ACTIVE tables; state consistent across dbml, summary and ledgers.
"""


# ====================================================
# Prompt and Response Helpers (Direct 1:1 match with dbNexus)
# ====================================================

def clean_dbml(dbml: str) -> str:
    dbml = dbml.strip()
    if dbml.startswith("```"):
        lines = [
            line for line in dbml.splitlines()
            if not line.strip().startswith("```")
        ]
        dbml = "\n".join(lines).strip()

    lines = [line.strip() for line in dbml.splitlines() if line.strip()]
    flattened = []
    table_lines = []

    for line in lines:
        table_lines.append(line)
        if line == "}":
            table_name = table_lines[0].rstrip("{").strip()
            columns = table_lines[1:-1]
            flattened.append(f"{table_name} {{{' '.join(columns)}}}")
            table_lines = []

    if table_lines:
        flattened.extend(table_lines)

    return " ".join(flattened).strip()


def validate_dbml(dbml: str) -> bool:
    if not dbml:
        return False
    normalized = clean_dbml(dbml)
    return "Table " in normalized and "{" in normalized and "}" in normalized


def normalize_llm_text(value: str) -> str:
    """Convert escaped control sequences from LLM output to real characters."""
    value = re.sub(r"\\+n", "\n", value)
    value = re.sub(r"\\+r", "\r", value)
    value = re.sub(r"\\+t", "\t", value)
    return value


def flatten_llm_text(value: str) -> str:
    """Normalize escaped sequences and collapse whitespace to spaces."""
    return " ".join(normalize_llm_text(value).split())


def parse_dbml_response(
    response: str, include_intent: bool = False
) -> dict[str, str]:
    response = response.strip()

    cleaned = response
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|dbml)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    payload = None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            payload = None

    if isinstance(payload, dict):
        dbml = payload.get("dbml")
        summary = payload.get("summary")
        explanation = payload.get("explanation")
        intent = payload.get("intent", "SCHEMA")

        if not all(
            isinstance(value, str) for value in (dbml, summary, explanation, intent)
        ):
            logger.warning(f"Invalid DBML response format from LLM: {payload}")
            raise ValueError("Invalid DBML response format from LLM.")

        if intent not in VALID_LLM_INTENTS:
            raise ValueError("Invalid LLM intent.")

        if not explanation.strip() or (intent == "SCHEMA" and not summary.strip()):
            raise ValueError("Invalid DBML response from LLM.")

        result = {
            "dbml_query": normalize_llm_text(dbml).strip(),
            "updated_summary": flatten_llm_text(summary).strip(),
            "explanation": flatten_llm_text(explanation),
        }
        if include_intent:
            result["intent"] = intent
        return result

    if "```" in response and not cleaned.startswith("{"):
        logger.warning(f"Invalid DBML response format: {response}")
        raise ValueError("Invalid DBML response format from LLM.")

    match = re.fullmatch(
        r"\s*(?:INTENT:\s*(?P<intent>[A-Z_]+)\s*)?"
        r"DBML:\s*(?P<dbml>.*?)\s*SUMMARY:\s*"
        r"(?P<summary>.*?)\s*EXPLANATION:\s*(?P<explanation>.*?)\s*",
        response,
        flags=re.DOTALL,
    )
    if not match:
        logger.warning(f"Invalid DBML response format: {response}")
        raise ValueError("Invalid DBML response format from LLM.")

    dbml = match.group("dbml").strip()
    summary = match.group("summary").strip()
    explanation = match.group("explanation").strip()

    if not summary or not explanation:
        raise ValueError("Invalid DBML response from LLM.")

    result = {
        "dbml_query": normalize_llm_text(dbml),
        "updated_summary": flatten_llm_text(summary),
        "explanation": flatten_llm_text(explanation),
    }
    if include_intent:
        result["intent"] = match.group("intent") or "SCHEMA"
    return result


def build_user_prompt(enable_summary: bool, summary: str, user_query: str, dbml: str = "") -> str:
    """Build standardized user prompt with full summary ledger synchronization matching dbNexus logic."""
    # Fallback: extract dbml if client embedded it inside summary and dbml was not provided
    if not (dbml or "").strip() and summary and "Current DBML:" in summary:
        parts = summary.split("Current DBML:", 1)
        summary = parts[0].strip()
        dbml = parts[1].strip()

    # Synchronize summary ledgers with active DBML:
    # Detect whether tables were deleted or renamed manually in the editor/canvas.
    if enable_summary and summary:
        active_table_blocks = re.findall(r"(?i)\btable\s+([a-zA-Z0-9_]+)\s*\{([^}]*)\}", dbml or "")
        active_tables = {tbl: " ".join(cols.split()) for tbl, cols in active_table_blocks}
        logger.info("Active tables in EXISTING_DBML: %s", list(active_tables.keys()))

        if active_tables or "Available Tables:" in summary:
            summary_tables = re.findall(
                r"-\s*([a-zA-Z0-9_]+):\s*(.*?)\s*Status:\s*([A-Za-z0-9_]+)",
                summary,
            )
            summary_map = {tbl: (" ".join(cols.rstrip(";").split()), status) for tbl, cols, status in summary_tables}

            missing_tables = [tbl for tbl in summary_map if tbl not in active_tables]
            new_active_tables = [tbl for tbl in active_tables if tbl not in summary_map]

            rename_map = {}
            for old_tbl in missing_tables:
                old_cols = summary_map[old_tbl][0]
                for new_tbl in new_active_tables:
                    if new_tbl not in rename_map.values():
                        new_cols = active_tables[new_tbl]
                        if old_cols and new_cols and old_cols == new_cols:
                            rename_map[old_tbl] = new_tbl
                            break

            def _sync_status(match):
                tbl = match.group(1)
                content = match.group(2).strip()
                status = match.group(3)
                if content and not content.endswith(";"):
                    content += ";"
                sep = f" {content} " if content else " "

                if tbl not in active_tables:
                    if tbl in rename_map:
                        new_name = rename_map[tbl]
                        logger.info("Table %s was renamed to %s in DBML. Marking as RENAMED.", tbl, new_name)
                        return f"- {tbl}:{sep}Status: RENAMED_TO_{new_name}"
                    elif status != "DROPPED" and not status.startswith("RENAMED"):
                        logger.info("Table %s is in summary but not in DBML. Marking as DROPPED.", tbl)
                        return f"- {tbl}:{sep}Status: DROPPED"
                    elif status == "DROPPED":
                        return f"- {tbl}:{sep}Status: DROPPED"
                elif tbl in active_tables:
                    if status != "ACTIVE":
                        logger.info("Table %s is in DBML. Marking as ACTIVE.", tbl)
                    return f"- {tbl}:{sep}Status: ACTIVE"
                return match.group(0)

            summary = re.sub(
                r"-\s*([a-zA-Z0-9_]+):\s*(.*?)\s*Status:\s*([A-Za-z0-9_]+)",
                _sync_status,
                summary,
            )

            unrecorded_active = [tbl for tbl in active_tables if tbl not in summary_map and tbl not in rename_map.values()]
            if unrecorded_active and "Available Tables:" in summary:
                logger.info("Found newly created active tables not in summary: %s", unrecorded_active)
                new_entries = "\n".join(f"- {tbl}: {active_tables[tbl]}; Status: ACTIVE" for tbl in unrecorded_active)
                summary = summary.replace("Available Tables:\n- None", f"Available Tables:\n{new_entries}")
                if "Available Tables:\n- None" not in summary:
                    summary = re.sub(r"(Available Tables:\n)", r"\1" + new_entries + "\n", summary, count=1)

    existing_summary = (summary or "").strip() if enable_summary else ""
    if not existing_summary:
        existing_summary = "NO PREVIOUS SUMMARY"

    return (
        f"EXISTING_SUMMARY:\n{existing_summary}\n\n"
        f"EXISTING_DBML:\n{dbml or 'No existing DBML.'}\n\n"
        f"CURRENT_QUERY:\n{user_query.strip()}\n\n"
        f"SUMMARY_ENABLED:\n{enable_summary}"
    )


def sql_to_dbml(sql: str) -> str:
    """Convert CREATE TABLE SQL statement to DBML format if needed."""
    if not sql or not sql.strip():
        return ""
    if "Table " in sql and "{" in sql:
        return sql.strip()

    m = re.search(r"CREATE\s+TABLE\s+([a-zA-Z0-9_]+)\s*\((.*)\)\s*;?", sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return sql.strip()

    tbl_name = m.group(1)
    body = m.group(2).strip()

    cols = []
    current = []
    paren_depth = 0
    for char in body:
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        if char == "," and paren_depth == 0:
            cols.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        cols.append("".join(current).strip())

    lines = [f"Table {tbl_name} {{"]
    for col_def in cols:
        tokens = col_def.split()
        if not tokens:
            continue
        c_name = tokens[0]
        c_type = tokens[1] if len(tokens) > 1 else "varchar"
        rest = " ".join(tokens[2:]).lower() if len(tokens) > 2 else ""
        attrs = []
        if "primary key" in rest:
            attrs.append("pk")
        if "unique" in rest:
            attrs.append("unique")
        if "not null" in rest:
            attrs.append("not null")
        attr_str = f" [{' '.join(attrs)}]" if attrs else ""
        lines.append(f"  {c_name} {c_type.lower()}{attr_str}")
    lines.append("}")
    return "\n".join(lines)


# ====================================================
# 1. Custom SLM Service
# ====================================================
class CustomSLMService:
    def __init__(self, model_dir: str = CUSTOM_MODEL_DIR):
        self.model_dir = model_dir
        onnx_model_path = os.path.join(model_dir, "model.onnx")
        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        topics_path = os.path.join(model_dir, "topics.json")

        if not os.path.exists(onnx_model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError(
                f"Custom model not found in {model_dir}. Run 'python src/train_slm.py' first!"
            )

        logger.info(f"Loading Custom SLM ONNX model from: {onnx_model_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_token_id = self.tokenizer.token_to_id("<eos>")
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_name = self.session.get_inputs()[0].name

        self.qa_pairs = []
        if os.path.exists(topics_path):
            with open(topics_path, "r", encoding="utf-8") as f:
                self.qa_pairs = json.load(f)
            logger.info(f"Loaded {len(self.qa_pairs)} fed knowledge topics for Custom SLM.")

    def _normalize(self, text: str) -> str:
        text = text.lower().replace("mitrah soft", "mitrahsoft")
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def _find_best_topic(self, query: str):
        norm_query = self._normalize(query)
        raw_tokens = set(norm_query.split())
        important_query = raw_tokens - IGNORE_WORDS

        if not important_query:
            return None

        best_topic = None
        highest_score = 0.0

        for topic in self.qa_pairs:
            norm_topic = self._normalize(topic)
            topic_raw_tokens = set(norm_topic.split())
            important_topic = topic_raw_tokens - IGNORE_WORDS

            foreign_entities = important_query - important_topic
            if foreign_entities:
                continue

            overlap = len(important_query & important_topic)
            if overlap == 0:
                continue

            query_coverage = overlap / len(important_query)
            union = len(important_query | important_topic)
            jaccard = overlap / union if union > 0 else 0.0
            score = (query_coverage * 0.7) + (jaccard * 0.3)

            if score > highest_score and score >= 0.4:
                highest_score = score
                best_topic = topic

        return best_topic

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        matched_topic = self._find_best_topic(prompt)
        if not matched_topic:
            return "I'm sorry, but I'm unable to assist with that."

        formatted_prompt = f"Prompt: {matched_topic}\nAnswer:"
        encoding = self.tokenizer.encode(formatted_prompt)
        prompt_len = len(encoding.ids)
        input_ids = list(encoding.ids)

        for _ in range(max_tokens):
            inputs_array = np.array([input_ids], dtype=np.int64)
            outputs = self.session.run(None, {self.input_name: inputs_array})
            logits = outputs[0][0, -1, :]
            next_token = int(np.argmax(logits))

            if next_token == self.eos_token_id:
                break

            input_ids.append(next_token)

        generated_tokens = input_ids[prompt_len:]
        decoded_text = self.tokenizer.decode(generated_tokens)
        clean_response = decoded_text.replace("<eos>", "").strip()

        if "Answer:" in clean_response:
            clean_response = clean_response.split("Answer:")[-1].strip()

        return clean_response if clean_response else "Not found"


# ====================================================
# 2. Qwen2.5-0.5B ONNX Service
# ====================================================
class QwenONNXService:
    def __init__(self, model_dir: str = QWEN_MODEL_DIR):
        self.model_dir = model_dir
        logger.info(f"Loading Qwen2.5-0.5B ONNX model from: {self.model_dir}")
        try:
            from transformers import AutoTokenizer
            from optimum.onnxruntime import ORTModelForCausalLM
        except Exception as e:
            raise ImportError(
                f"Failed to load dependencies for QwenONNXService: {e}. "
            ) from e

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        session_opts = ort.SessionOptions()
        session_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        self.model = ORTModelForCausalLM.from_pretrained(
            self.model_dir,
            subfolder="onnx",
            file_name="model_q4f16.onnx",
            session_options=session_opts,
        )
        logger.info("Qwen ONNX model loaded successfully.")

    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int = 150) -> str:
        messages = []
        if system_prompt:
            logger.info("Executed with system prompt")
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        logger.info("The onxx inner section exec---->")
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )


# ====================================================
# 3. Model Manager
# ====================================================
class ModelManager:
    def __init__(self):
        self._custom_service: CustomSLMService | None = None
        self._qwen_service: QwenONNXService | None = None

    @property
    def custom_service(self) -> CustomSLMService:
        if self._custom_service is None:
            self._custom_service = CustomSLMService()
        return self._custom_service

    @property
    def qwen_service(self) -> QwenONNXService:
        if self._qwen_service is None:
            self._qwen_service = QwenONNXService()
        return self._qwen_service

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 150,
        use_custom_onnx: bool = True,
    ) -> tuple[str, str]:
        if use_custom_onnx:
            logger.info("Routing request to: Custom SLM ONNX")
            response = self.custom_service.generate(prompt=prompt, max_tokens=max_tokens)
            return response, "custom-slm-onnx"
        else:
            logger.info("Routing request to: Qwen2.5-0.5B ONNX")
            response = self.qwen_service.generate(
                prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens
            )
            return response, "qwen2.5-0.5b-onnx"


_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


# ====================================================
# 4. LLM Calling & Token Estimation (Exact dbNexus structure)
# ====================================================

DEFAULT_OUTPUT_TOKEN_RESERVE = 300


class LLMCallResult:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def estimate_tokens(text: str, output_reserve: int = DEFAULT_OUTPUT_TOKEN_RESERVE) -> int:
    """Cheap provider-independent estimate: about four characters per token + output reserve."""
    input_estimate = max(1, math.ceil(len(text or "") / 4))
    return input_estimate + max(0, output_reserve)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    use_custom_onnx: bool = True,
    max_tokens: int = 250,
) -> LLMCallResult:
    """Make local ONNX LLM call passing system prompt and user prompt."""
    manager = get_model_manager()

    prompt_tokens = max(1, math.ceil((len(system_prompt or "") + len(user_prompt or "")) / 4))

    # If using custom SLM ONNX, match against user query intent; for Qwen ONNX, pass full prompt + system prompt
    query_to_model = user_prompt if not use_custom_onnx else (user_prompt.split('CURRENT_QUERY:\n')[-1].split('\n\nSUMMARY_ENABLED')[0] if 'CURRENT_QUERY:\n' in user_prompt else user_prompt)
    logger.info("Onxx model call started---->")
    content, model_name = manager.generate(
        prompt=query_to_model,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        use_custom_onnx=use_custom_onnx,
    )
    logger.info("Onxx model call ended---->")
    completion_tokens = max(1, math.ceil(len(content or "") / 4))
    total_tokens = prompt_tokens + completion_tokens

    logger.info(
        "LLM success model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
    )
    return LLMCallResult(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


# ====================================================
# Main generate_dbml implementation (Matches dbNexus structure)
# ====================================================

def generate_dbml(
    user_query: str,
    enable_summary: bool = False,
    summary: str = "",
    direct_model: str | None = None,
    llm: str | None = None,
    llm_api_key: str | None = None,
    base_url: str | None = None,
    dbml: str = "",
    use_custom_onnx: bool = True,
) -> dict:
    user_prompt = build_user_prompt(
        enable_summary=enable_summary,
        summary=summary,
        user_query=user_query,
        dbml=dbml,
    )
    estimated_tokens = estimate_tokens(DBML_SYSTEM_PROMPT + "\n" + user_prompt)

    # Call LLM with exact DBML_SYSTEM_PROMPT and user_prompt
    call = call_llm(
        system_prompt=DBML_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        use_custom_onnx=use_custom_onnx,
    )

    try:
        result = parse_dbml_response(call.content, include_intent=True)
        intent = result.pop("intent")

        if intent != "SCHEMA":
            result["dbml_query"] = ""
            result["updated_summary"] = summary
    except Exception:
        dbml_content = sql_to_dbml(call.content)
        if dbml and dbml.strip() and dbml_content:
            if dbml_content not in dbml:
                generated_dbml = f"{dbml.strip()}\n\n{dbml_content}"
            else:
                generated_dbml = dbml.strip()
        else:
            generated_dbml = dbml_content

        result = {
            "dbml_query": generated_dbml,
            "updated_summary": (
                f"Request Summary:\n- Applied: {user_query.strip()}\n\n"
                f"Available Tables:\n- {dbml_content}\n\n"
                f"Available Groups:\n- None\n\n"
                f"Available References:\n- None\n\n"
                f"Pending Rename:\n- None"
            ) if enable_summary else None,
            "explanation": f"Generated DBML schema for '{user_query}'.",
        }

    if not enable_summary:
        result["updated_summary"] = None
    result["token_used"] = call.total_tokens
    result["estimated_tokens"] = estimated_tokens
    result["model_used"] = "custom-slm-onnx" if use_custom_onnx else "qwen2.5-0.5b-onnx"
    return result
