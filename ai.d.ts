/**
 * TypeScript definitions for @oof2510/llmjs
 * Comprehensive type definitions for the LLM.js module
 */

import { ChatOpenAI } from "@langchain/openai";
import { BaseMessage, HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { Collection } from "mongodb";

declare module "@oof2510/llmjs" {
  /**
   * Media attachment types supported by the AI models
   */
  export interface AiAttachment {
    /** The media type being sent */
    type: "image" | "video";
    
    /** Remote URL that OpenRouter can fetch directly */
    url?: string;
    
    /** Raw file contents (Buffer or base64 string) */
    data?: string | Buffer;
    
    /** Optional MIME type for format hints */
    mimeType?: string;
    
    /** Explicit format override for video blobs (e.g., "mp4") */
    format?: string;
  }

  /**
   * Options for configuring the AiMemoryStore
   */
  export interface AiMemoryStoreOptions {
    /** Name of the MongoDB collection to use */
    collectionName?: string;
    
    /** MongoDB connection URI */
    uri: string;
    
    /** MongoDB database name */
    dbName: string;
  }

  /**
   * Stores conversation history in MongoDB for context-aware responses
   */
  export class AiMemoryStore {
    /**
     * Initializes the memory store with MongoDB connection details
     * @param options Configuration options for the memory store
     * @throws Error If required parameters are missing
     */
    constructor(options: AiMemoryStoreOptions);

    /**
     * Connects to MongoDB lazily and returns the collection
     * @returns Promise that resolves to the MongoDB collection
     */
    connect(): Promise<Collection>;

    /**
     * Closes the MongoDB connection
     * @returns Promise that resolves when connection is closed
     */
    disconnect(): Promise<void>;

    /**
     * Retrieves recent chat messages for context
     * @param chatId Identifier for the conversation
     * @param scope Context scope identifier
     * @param limit Maximum number of messages to retrieve (default: 10)
     * @returns Promise of message history array
     */
    getHistory(
      chatId: string | number,
      scope: string,
      limit?: number
    ): Promise<Array<{ role: string; content: string | Array<any> | Record<string, any> }>>;

    /**
     * Saves new messages and prunes old ones to maintain history size
     * @param chatId Identifier for the conversation
     * @param scope Context scope identifier
     * @param messages Array of messages to save
     * @returns Promise that resolves when messages are stored
     */
    appendMessages(
      chatId: string | number,
      scope: string,
      messages?: Array<{ role: string; content: string | Array<any> | Record<string, any> }>
    ): Promise<void>;

    /**
     * Clears all stored history for a conversation
     * @param chatId Identifier for the conversation
     * @param scope Context scope identifier
     * @returns Promise that resolves when history is cleared
     */
    clearHistory(chatId: string | number, scope: string): Promise<void>;
  }

  /**
   * Configuration options for the base AI class
   */
  export interface BaseAiOptions {
    /** API key for authentication */
    apiKey: string;
    
    /** Primary model to use (default: "meta-llama/llama-3.3-70b-instruct:free") */
    model?: string;
    
    /** Fallback models to try if primary fails */
    fallbackModels?: string[];
    
    /** Sampling temperature for generation (default: 0.7) */
    temperature?: number;
    
    /** Maximum number of tokens to generate (default: 1000) */
    maxTokens?: number;
    
    /** Custom headers to include with requests */
    defaultHeaders?: Record<string, string>;
    
    /** Request timeout in milliseconds (default: 20000) */
    requestTimeoutMs?: number;
  }

  /**
   * Converts mixed content into a consistent array format
   * @param content Input content of various types
   * @returns Normalized array of content pieces
   */
  export type ContentArray = Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string } }>;

  /**
   * Main AI class for interacting with language models via OpenRouter
   */
  export class Ai {
    /**
     * Initializes the AI helper with model preferences and request defaults
     * @param options Configuration options for the AI instance
     * @throws Error If API key is missing
     */
    constructor(options: BaseAiOptions);

    /**
     * Ensures content is in array format suitable for API consumption
     * @param content Input content to normalize
     * @returns Normalized content array
     */
    ensureContentArray(content: string | Array<any> | undefined): ContentArray;

    /**
     * Converts attachment to format compatible with AI APIs
     * @param attachment Attachment to normalize
     * @returns Normalized attachment object or null
     */
    normalizeAttachment(attachment: AiAttachment): Record<string, any> | null;

    /**
     * Converts data to base64 encoding
     * @param data Input data to encode
     * @returns Base64 string or null
     */
    toBase64(data: string | Buffer | undefined): string | null;

    /**
     * Creates data URI from binary data
     * @param data Input data for URI creation
     * @param mimeType MIME type for the data
     * @returns Data URI string or null
     */
    toDataUri(data: string | Buffer | undefined, mimeType: string): string | null;

    /**
     * Converts MIME type to format string
     * @param mimeType MIME type to convert
     * @returns Format string or undefined
     */
    mimeTypeToFormat(mimeType: string): string | undefined;

    /**
     * Prepares user message with attachments for API consumption
     * @param options Message construction options
     * @returns Object containing LangChain message and history content
     */
    buildUserMessagePayload(options?: {
      user?: any;
      attachments?: AiAttachment[];
    }): { message: HumanMessage | null; contentForHistory: string | Array<any> | Record<string, any> | null };

    /**
     * Retrieves a cached LangChain client for the specified model
     * @param model Model identifier
     * @returns Configured ChatOpenAI client
     */
    getClient(model: string): ChatOpenAI;

    /**
     * Constructs message array in format expected by LangChain
     * @param params Message construction parameters
     * @returns Array of formatted messages
     */
    buildMessages(params: {
      system?: string;
      messages?: BaseMessage[];
      user?: any;
      attachments?: AiAttachment[];
    }): BaseMessage[];

    /**
     * Extracts plain text from AI response
     * @param aiMessage Response message from AI
     * @returns Extracted text content
     */
    extractText(aiMessage: AIMessage | string | undefined): string;

    /**
     * Sends prompt to AI models with fallback support
     * @param options Prompt parameters
     * @returns Promise resolving to AI response text
     * @throws Error If no models are configured or all fail
     */
    ask(options?: {
      system?: string;
      user?: any;
      messages?: BaseMessage[];
      attachments?: AiAttachment[];
    }): Promise<string>;
  }

  /**
   * AI class with conversation history support
   */
  export class AiWithHistory extends Ai {
    /**
     * Initializes AI with history tracking capabilities
     * @param options History-enabled configuration
     * @throws Error If memory store is missing
     */
    constructor(options: {
      memoryStore: AiMemoryStore;
      memoryScope?: string;
      historyLimit?: number;
    } & BaseAiOptions);

    /**
     * Formats stored content for LangChain consumption
     * @param content Content to format
     * @returns Formatted content
     */
    formatStoredContent(content: string | Array<any> | Record<string, any>): string | Record<string, any>;

    /**
     * Executes query with history context and stores response
     * @param chatId Conversation identifier
     * @param options Query parameters
     * @returns Promise resolving to AI response
     */
    ask(
      chatId: string | number,
      options?: {
        system?: string;
        user?: any;
        attachments?: AiAttachment[];
      }
    ): Promise<string>;

    /**
     * Clears conversation history for a chat
     * @param chatId Conversation identifier
     * @returns Promise resolving when history is cleared
     */
    clear(chatId: string | number): Promise<void>;
  }

  /**
   * Groq version of the Ai class.
   * Same constructor shape, same ask() behavior, fallbacks included.
   */
  export class GroqAi {
    /**
     * Initializes the Groq AI helper with model preferences and request defaults
     * @param options Configuration options for the Groq AI instance
     * @throws Error If API key is missing
     */
    constructor(options: {
      apiKey: string;
      model?: string;
      fallbackModels?: string[];
      temperature?: number;
      maxTokens?: number;
      requestTimeoutMs?: number;
    });

    /**
     * Converts mixed message content into a consistent array
     * @param content Input content of various types
     * @returns Normalized array of content pieces
     */
    ensureContentArray(content: string | Array<any> | undefined): Array<{ type: "text"; text: string }>;

    /**
     * Turns an attachment descriptor into a Groq-friendly multimodal chunk
     * @param attachment Attachment to normalize
     * @returns Normalized attachment object or null
     */
    normalizeAttachment(attachment: AiAttachment): Record<string, any> | null;

    /**
     * Converts incoming blobs into base64 encoding
     * @param data Input data to encode
     * @returns Base64 string or null
     */
    toBase64(data: string | Buffer | undefined): string | null;

    /**
     * Creates data URI from binary data
     * @param data Input data for URI creation
     * @param mimeType MIME type for the data
     * @returns Data URI string or null
     */
    toDataUri(data: string | Buffer | undefined, mimeType: string): string | null;

    /**
     * Builds the outgoing user message with attachments for Groq
     * @param options Message construction options
     * @returns Object containing formatted content and history content
     */
    buildUserMessagePayload(options?: {
      user?: any;
      attachments?: AiAttachment[];
    }): { content: string | Array<any>; contentForHistory: string | Array<any> | null };

    /**
     * Extracts plain text from Groq response
     * @param resp Response from Groq API
     * @returns Extracted text content
     */
    extractText(resp: any): string;

    /**
     * Sends prompt to Groq models with fallback support
     * @param options Prompt parameters
     * @returns Promise resolving to AI response text
     * @throws Error If no models are configured or all fail
     */
    ask(options?: {
      system?: string;
      user?: any;
      messages?: BaseMessage[];
      attachments?: AiAttachment[];
    }): Promise<string>;

    /**
     * Transcribes an audio file using Groq Whisper
     * @param options Transcription parameters
     * @returns Promise resolving to transcribed text
     */
    transcribe(options: {
      file: string | Buffer | NodeJS.ReadableStream;
      model?: string;
      temperature?: number;
    }): Promise<string>;
  }

  /**
   * Groq version of AiWithHistory
   * Same API: ask(chatId, {...}), clear()
   */
  export class GroqAiWithHistory extends GroqAi {
    /**
     * Initializes AI with history tracking capabilities for Groq
     * @param options History-enabled configuration
     * @throws Error If memory store is missing
     */
    constructor(options: {
      memoryStore: AiMemoryStore;
      memoryScope?: string;
      historyLimit?: number;
    } & {
      apiKey: string;
      model?: string;
      fallbackModels?: string[];
      temperature?: number;
      maxTokens?: number;
      requestTimeoutMs?: number;
    });

    /**
     * Formats stored content for LangChain consumption
     * @param content Content to format
     * @returns Formatted content
     */
    formatStoredContent(content: string | Array<any> | Record<string, any>): string | Record<string, any>;

    /**
     * Executes query with history context and stores response for Groq
     * @param chatId Conversation identifier
     * @param options Query parameters
     * @returns Promise resolving to AI response
     */
    ask(
      chatId: string | number,
      options?: {
        system?: string;
        user?: any;
        attachments?: AiAttachment[];
      }
    ): Promise<string>;

    /**
     * Clears conversation history for a chat (Groq implementation)
     * @param chatId Conversation identifier
     * @returns Promise resolving when history is cleared
     */
    clear(chatId: string | number): Promise<void>;
  }

  /**
   * Mistral version of the Ai class.
   * Same constructor shape, same ask() behavior, fallbacks included.
   */
  export class MistralAi {
    /**
     * Initializes the Mistral AI helper with model preferences and request defaults
     * @param options Configuration options for the Mistral AI instance
     * @throws Error If API key is missing
     */
    constructor(options: {
      apiKey: string;
      model?: string;
      fallbackModels?: string[];
      temperature?: number;
      maxTokens?: number;
      requestTimeoutMs?: number;
    });

    /**
     * Converts mixed message content into a consistent array
     * @param content Input content of various types
     * @returns Normalized array of content pieces
     */
    ensureContentArray(content: string | Array<any> | undefined): Array<{ type: "text"; text: string }>;

    /**
     * Turns an attachment descriptor into a Mistral-friendly multimodal chunk
     * @param attachment Attachment to normalize
     * @returns Normalized attachment object or null
     */
    normalizeAttachment(attachment: AiAttachment): Record<string, any> | null;

    /**
     * Converts incoming blobs into base64 encoding
     * @param data Input data to encode
     * @returns Base64 string or null
     */
    toBase64(data: string | Buffer | undefined): string | null;

    /**
     * Creates data URI from binary data
     * @param data Input data for URI creation
     * @param mimeType MIME type for the data
     * @returns Data URI string or null
     */
    toDataUri(data: string | Buffer | undefined, mimeType: string): string | null;

    /**
     * Builds the outgoing user message with attachments for Mistral
     * @param options Message construction options
     * @returns Object containing formatted content and history content
     */
    buildUserMessagePayload(options?: {
      user?: any;
      attachments?: AiAttachment[];
    }): { content: string | Array<any>; contentForHistory: string | Array<any> | null };

    /**
     * Extracts plain text from Mistral response
     * @param resp Response from Mistral API
     * @returns Extracted text content
     */
    extractText(resp: any): string;

    /**
     * Sends prompt to Mistral models with fallback support
     * @param options Prompt parameters
     * @returns Promise resolving to AI response text
     * @throws Error If no models are configured or all fail
     */
    ask(options?: {
      system?: string;
      user?: any;
      messages?: BaseMessage[];
      attachments?: AiAttachment[];
    }): Promise<string>;

    /**
     * Transcribes an audio file using Mistral Voxtral
     * @param options Transcription parameters
     * @returns Promise resolving to transcribed text
     */
    transcribe(options: {
      file: string | Buffer | NodeJS.ReadableStream;
      model?: string;
      language?: string;
      timestamp_granularities?: string[];
    }): Promise<string>;

    /**
     * Classifies inputs using Mistral moderation models
     * @param inputs Text inputs to classify
     * @param options Classification parameters
     * @returns Promise resolving to classification results
     */
    classify(
      inputs: string | string[],
      options?: {
        model?: string;
        requestTimeoutMs?: number;
      }
    ): Promise<{ categories: Record<string, boolean>; scores: Record<string, number> }>;
  }

  /**
   * Mistral version of AiWithHistory
   * Same API: ask(chatId, {...}), clear()
   */
  export class MistralAiWithHistory extends MistralAi {
    /**
     * Initializes AI with history tracking capabilities for Mistral
     * @param options History-enabled configuration
     * @throws Error If memory store is missing
     */
    constructor(options: {
      memoryStore: AiMemoryStore;
      memoryScope?: string;
      historyLimit?: number;
    } & {
      apiKey: string;
      model?: string;
      fallbackModels?: string[];
      temperature?: number;
      maxTokens?: number;
      requestTimeoutMs?: number;
    });

    /**
     * Formats stored content for LangChain consumption
     * @param content Content to format
     * @returns Formatted content
     */
    formatStoredContent(content: string | Array<any> | Record<string, any>): string | Record<string, any>;

    /**
     * Executes query with history context and stores response for Mistral
     * @param chatId Conversation identifier
     * @param options Query parameters
     * @returns Promise resolving to AI response
     */
    ask(
      chatId: string | number,
      options?: {
        system?: string;
        user?: any;
        attachments?: AiAttachment[];
      }
    ): Promise<string>;

    /**
     * Clears conversation history for a chat (Mistral implementation)
     * @param chatId Conversation identifier
     * @returns Promise resolving when history is cleared
     */
    clear(chatId: string | number): Promise<void>;
  }
}