// Line-oriented SVGO service used by the evaluation harness.
// Input lines: {"svg": "...", "precision": 3 | null, "multipass": false}
// Output lines: {"svg": "..."} or {"error": "..."}
import { optimize } from "svgo";
import readline from "node:readline";

const lines = readline.createInterface({ input: process.stdin, terminal: false });
lines.on("line", (line) => {
  let reply;
  try {
    const request = JSON.parse(line);
    const options = { multipass: Boolean(request.multipass) };
    if (request.precision !== null && request.precision !== undefined) {
      options.floatPrecision = request.precision;
    }
    reply = { svg: optimize(request.svg, options).data };
  } catch (error) {
    reply = { error: String(error && error.message ? error.message : error) };
  }
  process.stdout.write(JSON.stringify(reply) + "\n");
});
