import { AgentRuntime, loadCharacterFromFile } from "@elizaos/core";

const character = await loadCharacterFromFile("characters/solana-trader.character.json");

const runtime = new AgentRuntime({
  character,
  modelProvider: "openai",
});

await runtime.start();

console.log(`✅ SolanaTrader arrancado correctamente`);
```

Luego en **Railway → Settings → Deploy → Start Command**:
```
bun agent.ts
