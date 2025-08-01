// src/telemetry/index.js
// Minimal OpenTelemetry Node tracer provider that prints spans to console.
// This is useful during development; swap the exporter in production.

import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import PrettyConsoleSpanExporter from './PrettyConsoleSpanExporter.js';

// Avoid double-registration when hot-reloading
if (!globalThis.__otel_provider_started) {
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(new PrettyConsoleSpanExporter()));
  provider.register();
  globalThis.__otel_provider_started = true;
  console.log('[Telemetry] OpenTelemetry ConsoleSpanExporter active');
} 