import { createAgentRuntime, loadCharacter } from "@elizaos/core";
import { readFileSync } from "fs";
import { resolve } from "path";

const characterPath = process.argv.find(arg => arg.startsWith("--character="))
  ?.split("=")[1] ?? "characters/solana-trader.character.json";

const character = JSON.parse(
  readFileSync(resolve(process.cwd(), characterPath), "utf-8")
);

const runtime = await createAgentRuntime({
  character,
  token: process.env.OPENAI_API_KEY!,
});

await runtime.start();

console.log(`✅ Agente ${character.name} iniciado correctamente`);
```

Luego en Railway → **Settings → Deploy → Start Command**:
```
bun start.ts --character=characters/solana-trader.character.json
