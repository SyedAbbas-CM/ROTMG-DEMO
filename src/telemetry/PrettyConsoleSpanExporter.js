import { ExportResultCode } from '@opentelemetry/core';

export default class PrettyConsoleSpanExporter {
  export(spans, resultCallback) {
    for (const span of spans) {
      const { name, duration, attributes } = span;
      const durMs = (duration[0] * 1e3 + duration[1] / 1e6).toFixed(1);
      const attr = JSON.stringify(attributes || {});
      // Print in compact single-line JSONL for easy post-processing
      console.log(JSON.stringify({ span: name, ms: +durMs, ...attributes }));
    }
    resultCallback({ code: ExportResultCode.SUCCESS });
  }

  shutdown() {
    return Promise.resolve();
  }
} 