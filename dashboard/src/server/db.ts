import "server-only";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

// Read-only handle on data/results.db, built by scripts/build-db.ts from results/*.json.
let db: DatabaseSync | undefined;

export function getDb(): DatabaseSync {
  db ??= new DatabaseSync(process.env.VIZWIZ_DB ?? join(process.cwd(), "data", "results.db"), {
    readOnly: true,
  });
  return db;
}

export function all<T>(sql: string, ...params: (string | number)[]): T[] {
  return getDb().prepare(sql).all(...params) as T[];
}

export function get<T>(sql: string, ...params: (string | number)[]): T | undefined {
  return getDb().prepare(sql).get(...params) as T | undefined;
}
