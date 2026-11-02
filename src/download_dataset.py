
from datasets import load_dataset

dataset = load_dataset(
    "neurondb/postgresql-llm",
    split="train"
)

ddl_keywords = (
    "CREATE TABLE",
    "ALTER TABLE",
    "DROP TABLE",
    "RENAME TABLE",
)

filtered = []

for row in dataset:
    sql = (row.get("sql") or "").strip()
    question = (row.get("question") or "").strip()

    normalized = sql.upper().lstrip()

    if not question or not sql:
        continue

    # Keep only the requested table-level DDL statements.
    if normalized.startswith(ddl_keywords):
        filtered.append({
            "question": question,
            "sql": sql,
        })

print("DDL examples:", len(filtered))

with open("sml_ddl_dataset.txt", "w", encoding="utf-8") as f:
    for row in filtered:
        # f.write("<bos>\n")
        f.write(f"Prompt: {row['question']}\n")
        f.write(f"Answer: {row['sql']}\n")
        f.write("<eos>\n\n")
