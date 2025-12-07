# @oof2510/llmjs

A wrapper for OpenRouter, Groq, and Mistral APIs with memory management, multimodal support, and fallback capabilities. This library provides an easy to use, unified interface for interacting with multiple AI providers while handling conversation history, media attachments, and error recovery.

## Installation

npm:
```bash
npm install @oof2510/llmjs
```

yarn:
```bash
yarn add @oof2510/llmjs
```

pnpm:
```bash
pnpm add @oof2510/llmjs
```

## Usage

### CommonJS

```javascript
const { Ai, AiMemoryStore, AiWithHistory, GroqAi, MistralAi } = require('@oof2510/llmjs');

// Basic AI usage
const ai = new Ai({
  apiKey: 'your-openrouter-api-key',
  model: 'meta-llama/llama-3.3-70b-instruct:free'
});

const response = await ai.ask({
  user: 'Hello, how are you?',
  system: 'You are a helpful assistant.'
});

console.log(response);
```

### ES6/TypeScript

```typescript
import { Ai, AiMemoryStore, AiWithHistory, GroqAi, MistralAi } from '@oof2510/llmjs';

// Basic AI usage
const ai = new Ai({
  apiKey: 'your-openrouter-api-key',
  model: 'meta-llama/llama-3.3-70b-instruct:free'
});

const response = await ai.ask({
  user: 'Hello, how are you?',
  system: 'You are a helpful assistant.'
});

console.log(response);
```

## Class Documentation

### Ai Class (OpenRouter)

#### Constructor

##### `new Ai(options)`

Creates a new AI instance for interacting with OpenRouter models.

- `options` (object, required):
  - `apiKey` (string, required): OpenRouter API key
  - `model` (string, optional): Default model (default: "meta-llama/llama-3.3-70b-instruct:free")
  - `fallbackModels` (string[], optional): Fallback models to try if primary fails
  - `temperature` (number, optional): Sampling temperature (0-1, default: 0.7)
  - `maxTokens` (number, optional): Maximum tokens to generate (default: 1000)
  - `defaultHeaders` (Record<string, string>, optional): Custom headers for requests
  - `requestTimeoutMs` (number, optional): Request timeout in milliseconds (default: 20000)
  - `firstToFinish` (boolean, optional): When true, sends the prompt to all configured models in parallel and returns the first successful response instead of trying models sequentially

#### Methods

##### `ask(options)`

Sends a prompt to AI models with fallback support.

- `options` (object, optional):
  - `system` (string, optional): System prompt
  - `user` (string|object|Array, optional): User message
  - `messages` (BaseMessage[], optional): Additional message history
  - `attachments` (AiAttachment[], optional): Media attachments
- **Returns:** Promise<string> - AI response text
- **Throws:** Error if no models configured or all fail

##### `ensureContentArray(content)`

Converts mixed content into a consistent array format.

- `content` (string|Array|undefined): Input content
- **Returns:** Array - Normalized content array

##### `normalizeAttachment(attachment)`

Converts attachment to format compatible with AI APIs.

- `attachment` (AiAttachment): Attachment to normalize
- **Returns:** Record<string, any> | null - Normalized attachment or null

##### `toBase64(data)`

Converts data to base64 encoding.

- `data` (string|Buffer|undefined): Input data
- **Returns:** string|null - Base64 string or null

##### `toDataUri(data, mimeType)`

Creates data URI from binary data.

- `data` (string|Buffer|undefined): Input data
- `mimeType` (string): MIME type
- **Returns:** string|null - Data URI or null

##### `mimeTypeToFormat(mimeType)`

Converts MIME type to format string.

- `mimeType` (string): MIME type
- **Returns:** string|undefined - Format string

##### `buildUserMessagePayload(options)`

Prepares user message with attachments for API consumption.

- `options` (object, optional):
  - `user` (any): User message
  - `attachments` (AiAttachment[]): Attachments
- **Returns:** { message: HumanMessage|null, contentForHistory: string|Array|Record<string, any>|null }

##### `getClient(model)`

Retrieves a cached LangChain client for the specified model.

- `model` (string): Model identifier
- **Returns:** ChatOpenAI - Configured client

##### `buildMessages(params)`

Constructs message array in format expected by LangChain.

- `params` (object):
  - `system` (string, optional): System prompt
  - `messages` (BaseMessage[], optional): Message history
  - `user` (any, optional): User message
  - `attachments` (AiAttachment[], optional): Attachments
- **Returns:** BaseMessage[] - Formatted messages

##### `extractText(aiMessage)`

Extracts plain text from AI response.

- `aiMessage` (AIMessage|string|undefined): AI response
- **Returns:** string - Extracted text

### AiWithHistory Class

Extends `Ai` with persistent memory capabilities.

#### Constructor

##### `new AiWithHistory(options)`

Creates a new AI instance with conversation history support.

- `options` (object):
  - `memoryStore` (AiMemoryStore, required): Memory store instance
  - `memoryScope` (string, optional): Memory namespace (default: "default")
  - `historyLimit` (number, optional): Messages to remember (default: 10)
  - `...other Ai options`

#### Methods

##### `ask(chatId, options)`

Fetches chat history, asks AI, and stores both user and bot messages.

- `chatId` (string|number, required): Conversation identifier
- `options` (object, optional):
  - `system` (string, optional): System prompt
  - `user` (any, optional): User message
  - `attachments` (AiAttachment[], optional): Attachments
- **Returns:** Promise<string> - AI response
- **Throws:** Error if chatId missing

##### `clear(chatId)`

Clears conversation history for a chat.

- `chatId` (string|number, required): Conversation identifier
- **Returns:** Promise<void>

### AiMemoryStore Class

MongoDB-based memory storage for conversation history.

#### Constructor

##### `new AiMemoryStore(options)`

Creates a new memory store for AI conversation history.

- `options` (object):
  - `uri` (string, required): MongoDB connection URI
  - `dbName` (string, required): Database name
  - `collectionName` (string, optional): Collection name (default: "ai_memory")
- **Throws:** Error if uri or dbName missing

#### Methods

##### `connect()`

Connects to MongoDB and returns the collection.

- **Returns:** Promise<Collection> - MongoDB collection

##### `disconnect()`

Closes the MongoDB connection.

- **Returns:** Promise<void>

##### `getHistory(chatId, scope, limit)`

Retrieves recent chat messages for context.

- `chatId` (string|number, required): Conversation identifier
- `scope` (string, required): Context scope
- `limit` (number, optional): Maximum messages to retrieve (default: 10)
- **Returns:** Promise<Array<{ role: string, content: string|Array|Record<string, any> }>> - Message history

##### `appendMessages(chatId, scope, messages)`

Saves new messages and prunes old ones.

- `chatId` (string|number, required): Conversation identifier
- `scope` (string, required): Context scope
- `messages` (Array<{ role: string, content: string|Array|Record<string, any> }>, optional): Messages to save
- **Returns:** Promise<void>

##### `clearHistory(chatId, scope)`

Clears all stored history for a conversation.

- `chatId` (string|number, required): Conversation identifier
- `scope` (string, required): Context scope
- **Returns:** Promise<void>

### GroqAi Class

Groq-specific AI implementation with same API as Ai class.

#### Constructor

##### `new GroqAi(options)`

Creates a new Groq AI instance.

- `options` (object):
  - `apiKey` (string, required): Groq API key
  - `model` (string, optional): Default model (default: "llama-3.1-70b-versatile")
  - `fallbackModels` (string[], optional): Fallback models
  - `temperature` (number, optional): Sampling temperature (default: 0.7)
  - `maxTokens` (number, optional): Maximum tokens (default: 1000)
  - `requestTimeoutMs` (number, optional): Request timeout (default: 20000)
  - `firstToFinish` (boolean, optional): When true, races all configured Groq models in parallel for each call and returns the first successful result
- **Throws:** Error if apiKey missing

#### Methods

##### `ask(options)`

Sends prompt to Groq models with fallback support.

- `options` (object, optional):
  - `system` (string, optional): System prompt
  - `user` (any, optional): User message
  - `messages` (BaseMessage[], optional): Message history
  - `attachments` (AiAttachment[], optional): Attachments
- **Returns:** Promise<string> - AI response
- **Throws:** Error if no models configured or all fail

##### `transcribe(options)`

Transcribes audio using Groq Whisper.

- `options` (object):
  - `file` (string|Buffer|ReadableStream, required): Audio file
  - `model` (string, optional): Transcription model (default: "whisper-large-v3-turbo")
  - `temperature` (number, optional): Sampling temperature (default: 0)
- **Returns:** Promise<string> - Transcribed text
- **Throws:** Error if file missing or transcription fails

### GroqAiWithHistory Class

Extends `GroqAi` with memory capabilities.

#### Constructor

##### `new GroqAiWithHistory(options)`

Creates a new Groq AI instance with history support.

- `options` (object):
  - `memoryStore` (AiMemoryStore, required): Memory store
  - `memoryScope` (string, optional): Memory scope (default: "default")
  - `historyLimit` (number, optional): History limit (default: 10)
  - `...other GroqAi options`
- **Throws:** Error if memoryStore missing

#### Methods

##### `ask(chatId, options)`

Executes query with history context and stores response.

- `chatId` (string|number, required): Conversation identifier
- `options` (object, optional):
  - `system` (string, optional): System prompt
  - `user` (any, optional): User message
  - `attachments` (AiAttachment[], optional): Attachments
- **Returns:** Promise<string> - AI response
- **Throws:** Error if chatId missing

##### `clear(chatId)`

Clears conversation history for a chat.

- `chatId` (string|number, required): Conversation identifier
- **Returns:** Promise<void>

### MistralAi Class

Mistral-specific AI implementation.

#### Constructor

##### `new MistralAi(options)`

Creates a new Mistral AI instance.

- `options` (object):
  - `apiKey` (string, required): Mistral API key
  - `model` (string, optional): Default model (default: "mistral-small-latest")
  - `fallbackModels` (string[], optional): Fallback models
  - `temperature` (number, optional): Sampling temperature (default: 0.7)
  - `maxTokens` (number, optional): Maximum tokens (default: 1000)
  - `requestTimeoutMs` (number, optional): Request timeout (default: 20000)
  - `firstToFinish` (boolean, optional): When true, races all configured Mistral models in parallel for each call and returns the first successful result
- **Throws:** Error if apiKey missing

#### Methods

##### `ask(options)`

Sends prompt to Mistral models with fallback support.

- `options` (object, optional):
  - `system` (string, optional): System prompt
  - `user` (any, optional): User message
  - `messages` (BaseMessage[], optional): Message history
  - `attachments` (AiAttachment[], optional): Attachments
- **Returns:** Promise<string> - AI response
- **Throws:** Error if no models configured or all fail

##### `transcribe(options)`

Transcribes audio using Mistral Voxtral.

- `options` (object):
  - `file` (string|Buffer|ReadableStream, required): Audio file
  - `model` (string, optional): Transcription model (default: "voxtral-mini-latest")
  - `language` (string, optional): Audio language
  - `timestamp_granularities` (string[], optional): Timestamp granularities
- **Returns:** Promise<string> - Transcribed text
- **Throws:** Error if file missing or transcription fails

##### `classify(inputs, options)`

Classifies inputs using Mistral moderation models.

- `inputs` (string|string[], required): Text inputs to classify
- `options` (object, optional):
  - `model` (string, optional): Moderation model (default: "mistral-moderation-latest")
  - `requestTimeoutMs` (number, optional): Request timeout
- **Returns:** Promise<{ categories: Record<string, boolean>, scores: Record<string, number> }> - Classification results
- **Throws:** Error if inputs missing or invalid

### MistralAiWithHistory Class

Extends `MistralAi` with memory capabilities.

#### Constructor

##### `new MistralAiWithHistory(options)`

Creates a new Mistral AI instance with history support.

- `options` (object):
  - `memoryStore` (AiMemoryStore, required): Memory store
  - `memoryScope` (string, optional): Memory scope (default: "default")
  - `historyLimit` (number, optional): History limit (default: 10)
  - `...other MistralAi options`
- **Throws:** Error if memoryStore missing

#### Methods

##### `ask(chatId, options)`

Executes query with history context and stores response.

- `chatId` (string|number, required): Conversation identifier
- `options` (object, optional):
  - `system` (string, optional): System prompt
  - `user` (any, optional): User message
  - `attachments` (AiAttachment[], optional): Attachments
- **Returns:** Promise<string> - AI response
- **Throws:** Error if chatId missing

##### `clear(chatId)`

Clears conversation history for a chat.

- `chatId` (string|number, required): Conversation identifier
- **Returns:** Promise<void>

## Attachments and Media Support

All AI classes support multimodal inputs through the `AiAttachment` interface:

```typescript
interface AiAttachment {
  type: "image" | "video";
  url?: string; // Remote URL
  data?: string | Buffer; // Raw file contents
  mimeType?: string; // MIME type hint
  format?: string; // Format override
}
```

### Image Attachments

```javascript
const imageAttachment = {
  type: 'image',
  // Option 1: Remote URL
  url: 'https://example.com/image.jpg',
  // Option 2: Inline data
  data: imageBuffer, // Buffer or base64 string
  mimeType: 'image/jpeg'
};
```

### Video Attachments

```javascript
const videoAttachment = {
  type: 'video',
  // Option 1: Remote URL
  url: 'https://example.com/video.mp4',
  // Option 2: Inline data
  data: videoBuffer,
  format: 'mp4'
};
```

### Usage Example

```javascript
const attachments = [imageAttachment, videoAttachment];

const response = await ai.ask({
  user: 'Describe this image and what happens in the video',
  attachments
});
```

## Configuration Options

### API Configuration

- **apiKey** (required): Authentication key for the AI provider
- **model** (optional): Primary model to use
- **fallbackModels** (optional): Models to try if primary fails
- **temperature** (optional): Controls randomness (0-1)
- **maxTokens** (optional): Limits response length
- **requestTimeoutMs** (optional): Request timeout in milliseconds
- **firstToFinish** (optional): If true, sends each request to all configured models in parallel and resolves with the first successful response (for both `ask` and, where supported, `transcribe`)

### Parallel racing with firstToFinish

When you configure multiple models, you can either try them one by one (sequential fallbacks, the default) or race them in parallel using `firstToFinish`:

```javascript
const ai = new Ai({
  apiKey: 'your-openrouter-api-key',
  model: 'primary-model',
  fallbackModels: ['fallback-1', 'fallback-2'],
  firstToFinish: true, // race all three models
});

const answer = await ai.ask({ user: 'Explain event loops in JS.' });

const groq = new GroqAi({
  apiKey: process.env.GROQ_API_KEY,
  fallbackModels: ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant'],
  firstToFinish: true,
});

// Also used for transcription: will race across configured transcription models
const text = await groq.transcribe({ file: 'audio.wav' });

const mistral = new MistralAi({
  apiKey: process.env.MISTRAL_API_KEY,
  fallbackModels: ['mistral-small-latest', 'mistral-large-latest'],
  firstToFinish: true,
});

const mistralAnswer = await mistral.ask({ user: 'Summarize this article.' });
```

### Memory Configuration

- **memoryStore** (required for history classes): MongoDB store instance
- **memoryScope** (optional): Namespace for memory (default: "default")
- **historyLimit** (optional): Messages to retain (default: 10)

### MongoDB Configuration

- **uri** (required): MongoDB connection URI
- **dbName** (required): Database name
- **collectionName** (optional): Collection name (default: "ai_memory")

## Fallback Models

The library automatically tries fallback models if the primary model fails:

```javascript
const ai = new Ai({
  apiKey: 'your-api-key',
  model: 'primary-model',
  fallbackModels: ['fallback1', 'fallback2', 'fallback3']
});
```

## Memory Management

### MongoDB Storage

```javascript
const memoryStore = new AiMemoryStore({
  uri: 'mongodb://localhost:27017',
  dbName: 'myapp',
  collectionName: 'ai_memory'
});
```

### Memory Operations

```javascript
// Store with memory
const ai = new AiWithHistory({
  apiKey: 'your-api-key',
  memoryStore,
  memoryScope: 'user123',
  historyLimit: 20
});

// Chat with memory
await ai.ask('chat123', { user: 'Remember this' });
await ai.ask('chat123', { user: 'What did I say before?' });

// Clear memory
await ai.clear('chat123');
```

## Error Handling

All methods throw errors for:

- Missing required parameters
- Invalid API keys
- Network failures
- Empty responses
- Timeout conditions

```javascript
try {
  const response = await ai.ask({ user: 'Hello!' });
} catch (error) {
  console.error('AI request failed:', error.message);
  console.error('Status code:', error.status);
  console.error('Original error:', error.original);
}
```

## License

MPL-2.0