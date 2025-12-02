# @oof2510/llmjs

An easy to use wrapper for OpenRouter, Groq, and Mistral APIs with memory management.

## Installation

```bash
npm install @oof2510/llmjs
```

or

```bash
yarn add @oof2510/llmjs
```

or

```bash
pnpm add @oof2510/llmjs
```

## Quick Start

### Basic AI Usage

```javascript
const { Ai } = require('@oof2510/llmjs');

const ai = new Ai({
  apiKey: 'your-openrouter-api-key',
  model: 'z-ai/glm-4.5-air:free'
});

const response = await ai.ask({
  user: 'Hello, how are you?',
  system: 'You are a helpful assistant.'
});

console.log(response);
```

### AI with Memory

```javascript
const { Ai, AiMemoryStore, AiWithHistory } = require('@oof2510/llmjs');

// Set up memory store
const memoryStore = new AiMemoryStore({
  uri: 'mongodb://localhost:27017',
  dbName: 'myapp',
  collectionName: 'ai_memory'
});

// Create AI with memory
const ai = new AiWithHistory({
  apiKey: 'your-openrouter-api-key',
  model: 'z-ai/glm-4.5-air:free',
  memoryStore,
  memoryScope: 'user123'
});

// Chat with memory
const response1 = await ai.ask('chat123', {
  user: 'My name is John'
});

const response2 = await ai.ask('chat123', {
  user: 'What is my name?'
}); // Will remember "John"

console.log(response2); // Should reference the name John
```

### Using Groq

```javascript
const { GroqAi } = require('@oof2510/llmjs');

const groq = new GroqAi({
  apiKey: 'your-groq-api-key',
  model: 'llama-3.1-70b-versatile'
});

const response = await groq.ask({
  user: 'Explain quantum computing in simple terms'
});
```

### Using Mistral

```javascript
const { MistralAi } = require('@oof2510/llmjs');

const mistral = new MistralAi({
  apiKey: 'your-mistral-api-key',
  model: 'mistral-small-latest'
});

const response = await mistral.ask({
  user: 'Write a haiku about programming'
});
```

## API Reference

### Ai Class (OpenRouter)

```javascript
const ai = new Ai({
  apiKey: 'your-openrouter-api-key', // Required
  model: 'z-ai/glm-4.5-air:free', // Default model
  fallbackModels: ['meta-llama/llama-3.3-70b-instruct:free'], // Fallback models
  temperature: 0.7, // 0-1
  maxTokens: 512, // Max response tokens
  defaultHeaders: {}, // Additional headers
  requestTimeoutMs: 20000 // Request timeout
});

const response = await ai.ask({
  system: 'You are a helpful assistant.', // Optional system prompt
  user: 'Hello!', // User message (string, object, or array)
  messages: [], // Additional message history
  attachments: [] // Media attachments
});
```

### AiWithHistory Class

Extends `Ai` with persistent memory.

```javascript
const ai = new AiWithHistory({
  apiKey: 'your-openrouter-api-key',
  memoryStore: memoryStoreInstance, // Required
  memoryScope: 'default', // Memory namespace
  historyLimit: 10, // Messages to remember
  // ... other Ai options
});

let chatId = '1234567890';

const response = await ai.ask(chatId, {
  user: 'Hello!',
  system: 'You are a helpful assistant.'
});

await ai.clear(chatId); // Clear memory for this chat
```

### AiMemoryStore Class

MongoDB-based memory storage.

```javascript
const memoryStore = new AiMemoryStore({
  uri: 'mongodb://localhost:27017', // Required
  dbName: 'myapp', // Required
  collectionName: 'ai_memory' // Optional, defaults to 'ai_memory'
});

let chatId = '1234567890';

// Manual memory operations
await memoryStore.appendMessages(chatId, scope, messages);
const history = await memoryStore.getHistory(chatId, scope, limit);
await memoryStore.clearHistory(chatId, scope);
await memoryStore.disconnect(); // Close connection
```

### GroqAi Class

```javascript
const groq = new GroqAi({
  apiKey: 'your-groq-api-key', // Required
  model: 'llama-3.1-70b-versatile', // Default model
  fallbackModels: ['llama-3.1-8b-instant'], // Fallback models
  temperature: 0.7,
  maxTokens: 512,
  requestTimeoutMs: 20000
});

// Same ask() API as Ai class
const response = await groq.ask({ user: 'Hello!' });

// Audio transcription
const transcription = await groq.transcribe({
  file: '/path/to/audio.mp3',
  model: 'whisper-large-v3-turbo'
});
```

### GroqAiWithHistory Class

Extends `GroqAi` with memory (same API as `AiWithHistory`).

### MistralAi Class

```javascript
const mistral = new MistralAi({
  apiKey: 'your-mistral-api-key', // Required
  model: 'mistral-small-latest', // Default model
  fallbackModels: ['mistral-medium-latest'], // Fallback models
  temperature: 0.7,
  maxTokens: 512,
  requestTimeoutMs: 20000
});

// Same ask() API as Ai class
const response = await mistral.ask({ user: 'Hello!' });

// Content classification
const classification = await mistral.classify(['inappropriate content'], {
  model: 'mistral-moderation-latest'
});
```

### MistralAiWithHistory Class

Extends `MistralAi` with memory (same API as `AiWithHistory`).

## Attachments

All AI classes support multimodal inputs:

```javascript
const attachments = [
  {
    type: 'image',
    url: 'https://example.com/image.jpg'
    // or
    data: imageBuffer, // Buffer or base64 string
    mimeType: 'image/jpeg'
  },
  {
    type: 'video',
    url: 'https://example.com/video.mp4'
    // or
    data: videoBuffer,
    format: 'mp4'
  }
];

const response = await ai.ask({
  user: 'Describe this image',
  attachments
});
```

## Error Handling

All methods will throw errors if:
- Required parameters are missing
- API keys are invalid
- Network requests fail
- All fallback models fail

```javascript
try {
  const response = await ai.ask({ user: 'Hello!' });
} catch (error) {
  console.error('AI request failed:', error.message);
}
```

## License

MPL-2.0