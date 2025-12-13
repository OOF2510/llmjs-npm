const { ChatOpenAI } = require("@langchain/openai");
const {
  HumanMessage,
  AIMessage,
  SystemMessage,
} = require("@langchain/core/messages");
const Groq = require("groq-sdk");
const { Mistral } = require("@mistralai/mistralai");
const { MongoClient } = require("mongodb");

/**
 * Default headers we send up with every OpenRouter call so the service knows who's pinging it.
 */
const DEFAULT_HEADERS = {
  "HTTP-Referer": "https://npmjs.com/package/@oof2510/llmjs",
  "X-Title": "LLM.js",
};

/**
 * @typedef {Object} AiAttachment
 * @property {"image"|"video"} type - The media type being sent up with the prompt.
 * @property {string} [url] - Remote URL that OpenRouter can fetch directly.
 * @property {string|Buffer} [data] - Raw file contents (Buffer or base64 string).
 * @property {string} [mimeType] - Optional MIME type so we can hint the format when sending inline data.
 * @property {string} [format] - Explicit format override for video blobs (e.g., "mp4").
 */

/**
 * Wraps a Mongo collection to stash little convo snippets so the bot remembers what folks said.
 */
class AiMemoryStore {
  /**
    * Sets up the memory store with the chosen collection name while deferring the actual connection.
    * @param {{ collectionName?: string, uri?: string, dbName?: string }} [options]
    */
   constructor({ collectionName = "ai_memory", uri, dbName } = {}) {
     if (!uri || !dbName) {
       throw new Error("uri and dbName are required for AiMemoryStore");
     }
     this.uri = uri;
     this.dbName = dbName;
     this.collectionName = collectionName;
     this.connectionPromise = null;
     this.client = null;
     this.collection = null;
     this.indexEnsured = false;
   }

  /**
   * Connects to Mongo lazily and hands back the collection once it's ready to use.
   * @returns {Promise<import("mongodb").Collection>}
   */
  async connect() {
    if (!this.connectionPromise) {
      this.connectionPromise = (async () => {
        this.client = new MongoClient(this.uri);
        await this.client.connect();
        const db = this.client.db(this.dbName);
        this.collection = db.collection(this.collectionName);
        if (!this.indexEnsured) {
          await this.collection.createIndex({
            scope: 1,
            chatId: 1,
            createdAt: 1,
          });
          this.indexEnsured = true;
        }
        return this.collection;
      })().catch((err) => {
        this.connectionPromise = null;
        throw err;
      });
    }

    await this.connectionPromise;
    return this.collection;
  }

  /**
   * Closes the Mongo connection so we aren't hoarding sockets when the bot shuts down.
   * @returns {Promise<void>}
   */
  async disconnect() {
    if (this.client) {
      await this.client.close();
      this.client = null;
      this.collection = null;
      this.connectionPromise = null;
      this.indexEnsured = false;
    }
  }

  /**
   * Pulls the latest chat messages for a scope so we can feed them back into the AI.
   * @param {string|number} chatId
   * @param {string} scope
   * @param {number} [limit]
   * @returns {Promise<Array<{role: string, content: string|Array|Record<string, any>}>>}
   */
  async getHistory(chatId, scope, limit = 10) {
    if (!chatId || !scope) return [];
    const collection = await this.connect();
    const normalizedChatId = String(chatId);
    const docs = await collection
      .find({ chatId: normalizedChatId, scope })
      .sort({ createdAt: -1 })
      .limit(limit)
      .toArray();
    return docs
      .reverse()
      .map((doc) => ({ role: doc.role, content: doc.content }))
      .filter((entry) => entry.role && entry.content);
  }

  /**
   * Saves fresh AI/user messages and trims the history so it doesn't blow up forever.
   * @param {string|number} chatId
   * @param {string} scope
   * @param {Array<{role: string, content: string|Array|Record<string, any>}>} messages
   * @returns {Promise<void>}
   */
  async appendMessages(chatId, scope, messages = []) {
    if (
      !chatId ||
      !scope ||
      !Array.isArray(messages) ||
      messages.length === 0
    ) {
      return;
    }
    const collection = await this.connect();
    const normalizedChatId = String(chatId);
    const docs = messages
      .filter((msg) => msg && msg.role && msg.content)
      .map((msg) => ({
        chatId: normalizedChatId,
        scope,
        role: msg.role,
        content: msg.content,
        createdAt: new Date(),
      }));

    if (docs.length === 0) return;

    await collection.insertMany(docs, { ordered: false }).catch((error) => {
      console.error("[AI Memory] Failed to append messages:", error);
    });

    // Keep only the latest 40 exchanges per chat scope to avoid unbounded growth
    const maxEntries = 80;
    collection
      .aggregate([
        { $match: { chatId: normalizedChatId, scope } },
        { $sort: { createdAt: -1 } },
        { $skip: maxEntries },
        { $project: { _id: 1 } },
      ])
      .toArray()
      .then((staleDocs) => {
        if (staleDocs.length) {
          const ids = staleDocs.map((doc) => doc._id);
          collection.deleteMany({ _id: { $in: ids } }).catch((err) => {
            console.error("[AI Memory] Failed to trim history:", err);
          });
        }
      })
      .catch((err) => {
        console.error("[AI Memory] Failed to schedule trim:", err);
      });
  }

  /**
   * Nukes any remembered lines for a chat scope when someone wants a clean slate.
   * @param {string|number} chatId
   * @param {string} scope
   * @returns {Promise<void>}
   */
  async clearHistory(chatId, scope) {
    if (!chatId || !scope) return;
    const collection = await this.connect();
    await collection.deleteMany({ chatId: String(chatId), scope });
    console.log(`[AI Memory] History cleared for ${chatId} in scope ${scope}`);
  }
}


/**
 * High-level wrapper for firing prompts at OpenRouter while handling fallbacks.
 */
class Ai {
  /**
    * Spins up the AI helper with model preferences and request defaults.
    * @param {{
    *  apiKey: string,
    *  model?: string,
    *  fallbackModels?: string[],
    *  temperature?: number,
    *  maxTokens?: number,
    *  defaultHeaders?: Record<string, string>,
    *  requestTimeoutMs?: number
    *  firstToFinish?: boolean
    * }} [options]
    */
    constructor({
      apiKey,
      model = "meta-llama/llama-3.3-70b-instruct:free",
      fallbackModels = [],
      temperature = 0.7,
      maxTokens = 1000,
      defaultHeaders = {},
      requestTimeoutMs = 20000,
      firstToFinish = false,
    } = {}) {
     if (!apiKey) {
       throw new Error("apiKey is required for Ai");
     }
     this.apiKey = apiKey;
     this.models = [model, ...fallbackModels].filter(Boolean);
     this.temperature = temperature;
     this.maxTokens = maxTokens;
     this.defaultHeaders = { ...DEFAULT_HEADERS, ...defaultHeaders };
     this.clientCache = new Map();
     this.lastUsedModel = null;
     // Per-request timeout (ms) used when invoking each model. If a model
     // doesn't respond within this window we try the next configured model.
     // Default: 20000ms (20s). A value of 0 disables the timeout.
     this.requestTimeoutMs = Number(requestTimeoutMs) || 0;
     this.firstToFinish = firstToFinish;
   }

  /**
   * Converts mixed message content into a consistent array so we can tack on attachments.
   * @param {string|Array|undefined} content
   * @returns {Array}
   */
  ensureContentArray(content) {
    if (Array.isArray(content)) {
      return [...content];
    }
    if (typeof content === "string" && content.length) {
      return [{ type: "text", text: content }];
    }
    return [];
  }

  /**
   * Turns an attachment descriptor into a LangChain-friendly multimodal chunk.
   * @param {AiAttachment} attachment
   * @returns {Record<string, any>|null}
   */
  normalizeAttachment(attachment) {
    if (!attachment || !attachment.type) return null;
    const type = String(attachment.type).toLowerCase();
    const { url, data, mimeType, format } = attachment;

    if (type === "image") {
      if (url) {
        return { type: "image_url", image_url: { url } };
      }

      const dataUri = this.toDataUri(data, mimeType || "image/png");
      if (!dataUri) return null;
      return { type: "image_url", image_url: { url: dataUri } };
    }

    if (type === "video") {
      if (url) {
        return { type: "input_video", video: { url } };
      }

      const encoded = this.toBase64(data);
      if (!encoded) return null;
      const payload = { data: encoded };
      const resolvedFormat =
        format || (mimeType ? this.mimeTypeToFormat(mimeType) : undefined);
      if (resolvedFormat) {
        payload.format = resolvedFormat;
      }
      return { type: "input_video", video: payload };
    }

    return null;
  }

  /**
   * Converts incoming blobs into base64 so the API can ingest them inline.
   * @param {string|Buffer|undefined} data
   * @returns {string|null}
   */
  toBase64(data) {
    if (!data) return null;
    if (Buffer.isBuffer(data)) {
      return data.toString("base64");
    }
    if (typeof data === "string") {
      const trimmed = data.trim();
      if (!trimmed) return null;
      if (trimmed.startsWith("data:")) {
        const base64Part = trimmed.substring(trimmed.indexOf(",") + 1);
        return base64Part ? base64Part.trim() : null;
      }
      const base64Like = trimmed.replace(/\s+/g, "");
      if (/^[A-Za-z0-9+/]+={0,2}$/.test(base64Like)) {
        return base64Like;
      }
      return Buffer.from(trimmed).toString("base64");
    }
    return null;
  }

  /**
   * Shapes data blobs into data URIs when the API expects them that way (mostly for images).
   * @param {string|Buffer|undefined} data
   * @param {string} mimeType
   * @returns {string|null}
   */
  toDataUri(data, mimeType) {
    if (!data) return null;
    if (typeof data === "string" && data.trim().startsWith("data:")) {
      return data.trim();
    }
    const encoded = this.toBase64(data);
    if (!encoded) return null;
    return `data:${mimeType};base64,${encoded}`;
  }

  /**
   * Attempts to turn a MIME type into a bare format string.
   * @param {string} mimeType
   * @returns {string|undefined}
   */
  mimeTypeToFormat(mimeType) {
    if (!mimeType) return undefined;
    const [, subtype] = mimeType.split("/");
    if (!subtype) return undefined;
    return subtype.split(";")[0] || undefined;
  }

  /**
   * Builds the outgoing user message and hands back a representation suitable for persistence.
   * @param {{ user?: any, attachments?: AiAttachment[] }} [options]
   * @returns {{ message: HumanMessage|null, contentForHistory: string|Array|Record<string, any>|null }}
   */
  buildUserMessagePayload({ user, attachments = [] } = {}) {
    const normalizedAttachments = Array.isArray(attachments)
      ? attachments
          .map((attachment) => this.normalizeAttachment(attachment))
          .filter(Boolean)
      : [];

    if (user instanceof HumanMessage) {
      if (!normalizedAttachments.length) {
        return { message: user, contentForHistory: user.content };
      }
      const contentPieces = this.ensureContentArray(user.content).concat(
        normalizedAttachments,
      );
      const rebuilt = new HumanMessage({
        content: contentPieces,
        additional_kwargs: user.additional_kwargs,
        name: user.name,
      });
      return { message: rebuilt, contentForHistory: contentPieces };
    }

    if (Array.isArray(user)) {
      const contentPieces = [...user, ...normalizedAttachments];
      if (!contentPieces.length) {
        return { message: null, contentForHistory: null };
      }
      return {
        message: new HumanMessage({ content: contentPieces }),
        contentForHistory: contentPieces,
      };
    }

    if (
      user &&
      typeof user === "object" &&
      user !== null &&
      "content" in user
    ) {
      const baseContent = this.ensureContentArray(user.content).concat(
        normalizedAttachments,
      );
      if (!baseContent.length) {
        return { message: null, contentForHistory: null };
      }
      return {
        message: new HumanMessage({
          ...user,
          content: baseContent,
        }),
        contentForHistory: baseContent,
      };
    }

    const trimmedUser = typeof user === "string" ? user : undefined;

    if (normalizedAttachments.length === 0) {
      if (typeof trimmedUser === "string" && trimmedUser.length) {
        return {
          message: new HumanMessage(trimmedUser),
          contentForHistory: trimmedUser,
        };
      }
      return { message: null, contentForHistory: null };
    }

    const contentPieces = [];
    if (typeof trimmedUser === "string" && trimmedUser.length) {
      contentPieces.push({ type: "text", text: trimmedUser });
    }
    contentPieces.push(...normalizedAttachments);

    if (!contentPieces.length) {
      return { message: null, contentForHistory: null };
    }

    return {
      message: new HumanMessage({ content: contentPieces }),
      contentForHistory: contentPieces,
    };
  }

  /**
    * Returns a cached LangChain client for the given model so we aren't rebuilding it each ask.
    * @param {string} model
    * @returns {import("@langchain/openai").ChatOpenAI}
    */
   getClient(model) {
     if (!this.clientCache.has(model)) {
       const client = new ChatOpenAI({
         apiKey: this.apiKey,
         model,
         temperature: this.temperature,
         maxTokens: this.maxTokens,
         configuration: {
           baseURL: "https://openrouter.ai/api/v1",
           defaultHeaders: this.defaultHeaders,
         },
       });
       this.clientCache.set(model, client);
     }
     return this.clientCache.get(model);
   }

  /**
   * Shapes the message payload so LangChain gets the context in the order it expects.
   * @param {{ system?: string, messages?: Array<import("@langchain/core/messages").BaseMessage>, user?: any, attachments?: AiAttachment[] }} params
   * @returns {Array<import("@langchain/core/messages").BaseMessage>}
   */
  buildMessages({ system, messages = [], user, attachments = [] }) {
    const payload = [];
    if (system) {
      payload.push(new SystemMessage(system));
    }
    if (messages.length) {
      payload.push(...messages);
    }
    const { message } = this.buildUserMessagePayload({ user, attachments });
    if (message) {
      payload.push(message);
    }
    return payload;
  }

  /**
   * Normalizes the AI response content into a clean string no matter the underlying format.
   * @param {import("@langchain/core/messages").AIMessage | string | undefined} aiMessage
   * @returns {string}
   */
  extractText(aiMessage) {
    if (!aiMessage) return "";
    if (typeof aiMessage.content === "string") {
      return aiMessage.content;
    }
    if (Array.isArray(aiMessage.content)) {
      return aiMessage.content
        .map((chunk) => {
          if (typeof chunk === "string") return chunk;
          if (typeof chunk?.text === "string") return chunk.text;
          if (typeof chunk?.content === "string") return chunk.content;
          return "";
        })
        .join("\n")
        .trim();
    }
    return "";
  }

  /**
   * Sends a prompt to the configured models, trying fallbacks until one answers.
   * @param {{ system?: string, user?: any, messages?: Array<import("@langchain/core/messages").BaseMessage>, attachments?: AiAttachment[] }} [options]
   * @returns {Promise<string>}
   */
  async ask({ system, user, messages = [], attachments = [] } = {}) {
    if (!this.models.length) {
      throw new Error("No AI models configured");
    }

    const builtMessages = this.buildMessages({
      system,
      user,
      messages,
      attachments,
    });

    if (!this.firstToFinish || this.models.length === 1) {
      let lastError;
      for (const model of this.models) {
        try {
          const client = this.getClient(model);

          let response;
          if (this.requestTimeoutMs > 0) {
            let timer;
            const work = client.invoke(builtMessages).then((res) => {
              if (timer) clearTimeout(timer);
              return res;
            });

            const timeoutPromise = new Promise((_, reject) => {
              timer = setTimeout(() => {
                reject(
                  new Error(
                    `Model ${model} timed out after ${this.requestTimeoutMs}ms`,
                  ),
                );
              }, this.requestTimeoutMs);
            });

            response = await Promise.race([work, timeoutPromise]);
          } else {
            response = await client.invoke(builtMessages);
          }

          const text = this.extractText(response)?.trim();
          if (!text) {
            throw new Error(`Empty response from ${model}`);
          }
          this.lastUsedModel = model;
          return text;
        } catch (error) {
          lastError = error;
          console.error(`[AI] ${model} failed:`, error?.message || error);
        }
      }

      throw lastError || new Error("All AI models failed");
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      let remaining = this.models.length;
      let lastError;

      for (const model of this.models) {
        (async () => {
          try {
            const client = this.getClient(model);

            let response;
            if (this.requestTimeoutMs > 0) {
              let timer;
              const work = client.invoke(builtMessages).then((res) => {
                if (timer) clearTimeout(timer);
                return res;
              });

              const timeoutPromise = new Promise((_, reject) => {
                timer = setTimeout(() => {
                  reject(
                    new Error(
                      `Model ${model} timed out after ${this.requestTimeoutMs}ms`,
                    ),
                  );
                }, this.requestTimeoutMs);
              });

              response = await Promise.race([work, timeoutPromise]);
            } else {
              response = await client.invoke(builtMessages);
            }

            const text = this.extractText(response)?.trim();
            if (!text) {
              throw new Error(`Empty response from ${model}`);
            }

            if (!settled) {
              settled = true;
              this.lastUsedModel = model;
              resolve(text);
            }
          } catch (error) {
            lastError = error;
            console.error(`[AI] ${model} failed:`, error?.message || error);
          } finally {
            remaining -= 1;
            if (!settled && remaining === 0) {
              settled = true;
              reject(lastError || new Error("All AI models failed"));
            }
          }
        })();
      }
    });
  }
}

/**
 * AI helper that remembers per-chat history so replies can reference older messages.
 */
class AiWithHistory extends Ai {
  /**
   * Rehydrates stored content into something LangChain message constructors understand.
   * @param {string|Array|Record<string, any>} content
   * @returns {string|Record<string, any>}
   */
  formatStoredContent(content) {
    if (Array.isArray(content)) {
      return { content };
    }
    if (content && typeof content === "object" && "content" in content) {
      return content;
    }
    return content;
  }

  /**
    * Boots the AI helper while wiring in a memory store for contextual answers.
    * @param {{
    *  memoryStore: AiMemoryStore,
    *  memoryScope?: string,
    *  historyLimit?: number
    * }} [options]
    */
   constructor({
     memoryStore,
     memoryScope = "default",
     historyLimit = 10,
     ...options
   } = {}) {
     if (!memoryStore) {
       throw new Error("memoryStore is required for AiWithHistory");
     }
     super(options);
     this.memoryStore = memoryStore;
     this.memoryScope = memoryScope;
     this.historyLimit = historyLimit;
   }

  /**
   * Fetches chat history, asks the AI, and stores both the user and bot messages.
   * @param {string|number} chatId
   * @param {{ system?: string, user?: any, attachments?: AiAttachment[] }} [options]
   * @returns {Promise<string>}
   */
  async ask(chatId, { system, user, attachments = [] } = {}) {
    if (!chatId) {
      throw new Error("chatId is required for AiWithHistory");
    }

    const historyEntries = await this.memoryStore.getHistory(
      chatId,
      this.memoryScope,
      this.historyLimit,
    );

    const formattedHistory = historyEntries.map((entry) => {
      const payload = this.formatStoredContent(entry.content);
      return entry.role === "assistant"
        ? new AIMessage(payload)
        : new HumanMessage(payload);
    });

    const { contentForHistory } = this.buildUserMessagePayload({
      user,
      attachments,
    });

    const response = await super.ask({
      system,
      user,
      attachments,
      messages: formattedHistory,
    });

    const toPersist = [];
    if (contentForHistory !== null && contentForHistory !== undefined) {
      toPersist.push({ role: "user", content: contentForHistory });
    }
    if (response) toPersist.push({ role: "assistant", content: response });

    if (toPersist.length) {
      this.memoryStore
        .appendMessages(chatId, this.memoryScope, toPersist)
        .catch((err) => {
          console.error("[AI Memory] Failed to persist chat:", err);
        });
    }

    return response;
  }

  /**
   * Clears any stored memory for the chat so the AI forgets the convo trail.
   * @param {string|number} chatId
   * @returns {Promise<void>}
   */
  async clear(chatId) {
    await this.memoryStore.clearHistory(chatId, this.memoryScope);
  }
}

/**
 * Groq version of the Ai class.
 * Same constructor shape, same ask() behavior, fallbacks included.
 */
class GroqAi {
  /**
   * Sets up the Groq helper with model preferences and request defaults.
   * @param {{
   *  apiKey: string,
   *  model?: string,
   *  fallbackModels?: string[],
   *  temperature?: number,
   *  maxTokens?: number,
   *  requestTimeoutMs?: number,
   *  firstToFinish?: boolean
   * }} [options]
   */
  constructor({
     apiKey,
     model = "llama-3.1-70b-versatile",
     fallbackModels = [],
     temperature = 0.7,
     maxTokens = 1000,
     requestTimeoutMs = 20000,
     firstToFinish = false,
   } = {}) {
     if (!apiKey) {
       throw new Error("apiKey is required for GroqAi");
     }
     this.models = [model, ...fallbackModels].filter(Boolean);
     this.temperature = temperature;
     this.maxTokens = maxTokens;
     this.requestTimeoutMs = Number(requestTimeoutMs) || 0;
     this.client = new Groq({ apiKey });
     this.lastUsedModel = null;
     this.firstToFinish = firstToFinish;
   }

  /**
   * Converts mixed message content into a consistent array so we can tack on attachments.
   * @param {string|Array|undefined} content
   * @returns {Array}
   */
  ensureContentArray(content) {
    if (Array.isArray(content)) {
      return [...content];
    }
    if (typeof content === "string" && content.length) {
      return [{ type: "text", text: content }];
    }
    return [];
  }

  /**
   * Turns an attachment descriptor into a Groq-friendly multimodal chunk.
   * @param {AiAttachment} attachment
   * @returns {Record<string, any>|null}
   */
  normalizeAttachment(attachment) {
    if (!attachment || !attachment.type) return null;
    const type = String(attachment.type).toLowerCase();
    const { url, data, mimeType, format } = attachment;

    if (type === "image") {
      if (url) {
        return { type: "image_url", image_url: { url } };
      }

      const dataUri = this.toDataUri(data, mimeType || "image/png");
      if (!dataUri) return null;
      return { type: "image_url", image_url: { url: dataUri } };
    }

    // Groq doesn't support video attachments yet, but we keep the structure for consistency
    if (type === "video") {
      console.warn("[GroqAI] Video attachments not supported by Groq");
      return null;
    }

    return null;
  }

  /**
   * Converts incoming blobs into base64 so the API can ingest them inline.
   * @param {string|Buffer|undefined} data
   * @returns {string|null}
   */
  toBase64(data) {
    if (!data) return null;
    if (Buffer.isBuffer(data)) {
      return data.toString("base64");
    }
    if (typeof data === "string") {
      const trimmed = data.trim();
      if (!trimmed) return null;
      if (trimmed.startsWith("data:")) {
        const base64Part = trimmed.substring(trimmed.indexOf(",") + 1);
        return base64Part ? base64Part.trim() : null;
      }
      const base64Like = trimmed.replace(/\s+/g, "");
      if (/^[A-Za-z0-9+/]+={0,2}$/.test(base64Like)) {
        return base64Like;
      }
      return Buffer.from(trimmed).toString("base64");
    }
    return null;
  }

  /**
   * Shapes data blobs into data URIs when the API expects them that way (mostly for images).
   * @param {string|Buffer|undefined} data
   * @param {string} mimeType
   * @returns {string|null}
   */
  toDataUri(data, mimeType) {
    if (!data) return null;
    if (typeof data === "string" && data.trim().startsWith("data:")) {
      return data.trim();
    }
    const encoded = this.toBase64(data);
    if (!encoded) return null;
    return `data:${mimeType};base64,${encoded}`;
  }

  /**
   * Builds the outgoing user message with attachments for Groq.
   * @param {{ user?: any, attachments?: AiAttachment[] }} [options]
   * @returns {{ content: string|Array, contentForHistory: string|Array|null }}
   */
  buildUserMessagePayload({ user, attachments = [] } = {}) {
    const normalizedAttachments = Array.isArray(attachments)
      ? attachments
          .map((attachment) => this.normalizeAttachment(attachment))
          .filter(Boolean)
      : [];

    const baseContent = this.ensureContentArray(
      typeof user === "string" ? user : user?.content,
    );

    const contentPieces = [...baseContent, ...normalizedAttachments];

    if (!contentPieces.length) {
      return { content: "", contentForHistory: null };
    }

    // For Groq, if we have attachments, we need to use the content array format
    // Otherwise, we can use a simple string
    const finalContent =
      normalizedAttachments.length > 0
        ? contentPieces
        : typeof user === "string"
          ? user
          : user?.content || "";

    return {
      content: finalContent,
      contentForHistory:
        normalizedAttachments.length > 0 ? contentPieces : finalContent,
    };
  }

  extractText(resp) {
    try {
      return resp.choices?.[0]?.message?.content?.trim() || "";
    } catch {
      return "";
    }
  }

  async ask({ system, user, messages = [], attachments = [] } = {}) {
    if (!this.models.length) throw new Error("No Groq models configured");

    const groqMessages = [];
    if (system) groqMessages.push({ role: "system", content: system });

    if (messages.length) {
      groqMessages.push(
        ...messages.map((m) => {
          // Handle plain objects that already have role/content
          if (m.role && m.content) {
            return {
              role: m.role,
              content:
                typeof m.content === "string"
                  ? m.content
                  : JSON.stringify(m.content),
            };
          }
          // Handle LangChain message objects
          if (typeof m._getType === "function") {
            return {
              role: m._getType() === "ai" ? "assistant" : "user",
              content:
                typeof m.content === "string"
                  ? m.content
                  : JSON.stringify(m.content),
            };
          }
          // Fallback for unknown format
          return {
            role: "user",
            content: JSON.stringify(m),
          };
        }),
      );
    }

    // Handle user message with attachments
    if (user || attachments.length > 0) {
      const { content } = this.buildUserMessagePayload({ user, attachments });
      groqMessages.push({ role: "user", content });
    }

    if (!this.firstToFinish || this.models.length === 1) {
      let lastError;
      for (const model of this.models) {
        try {
          const work = this.client.chat.completions.create({
            model,
            messages: groqMessages,
            temperature: this.temperature,
            max_tokens: this.maxTokens,
          });

          const resp =
            this.requestTimeoutMs > 0
              ? await Promise.race([
                  work,
                  new Promise((_, reject) =>
                    setTimeout(
                      () =>
                        reject(
                          new Error(
                            `Model ${model} timed out after ${this.requestTimeoutMs}ms`,
                          ),
                        ),
                      this.requestTimeoutMs,
                    ),
                  ),
                ])
              : await work;

          const text = this.extractText(resp);
          if (!text) throw new Error(`Empty response from ${model}`);

          this.lastUsedModel = model;
          return text;
        } catch (err) {
          lastError = err;
          console.error(`[GroqAI] ${model} failed:`, err?.message || err);
        }
      }

      throw lastError || new Error("All Groq models failed");
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      let remaining = this.models.length;
      let lastError;

      for (const model of this.models) {
        (async () => {
          try {
            const work = this.client.chat.completions.create({
              model,
              messages: groqMessages,
              temperature: this.temperature,
              max_tokens: this.maxTokens,
            });

            const resp =
              this.requestTimeoutMs > 0
                ? await Promise.race([
                    work,
                    new Promise((_, reject) =>
                      setTimeout(
                        () =>
                          reject(
                            new Error(
                              `Model ${model} timed out after ${this.requestTimeoutMs}ms`,
                            ),
                          ),
                        this.requestTimeoutMs,
                      ),
                    ),
                  ])
                : await work;

            const text = this.extractText(resp);
            if (!text) throw new Error(`Empty response from ${model}`);

            if (!settled) {
              settled = true;
              this.lastUsedModel = model;
              resolve(text);
            }
          } catch (err) {
            lastError = err;
            console.error(`[GroqAI] ${model} failed:`, err?.message || err);
          } finally {
            remaining -= 1;
            if (!settled && remaining === 0) {
              settled = true;
              reject(lastError || new Error("All Groq models failed"));
            }
          }
        })();
      }
    });
  }

  /**
   * Transcribes an audio file using Groq Whisper.
   * @param {Object} options
   * @param {string|Buffer|fs.ReadStream} options.file - Path, buffer, or stream
   * @param {string} [options.model="whisper-large-v3-turbo"]
   * @param {number} [options.temperature=0]
   * @returns {Promise<string>}
   */
  async transcribe({
    file,
    model = "whisper-large-v3-turbo",
    temperature = 0,
  } = {}) {
    if (!file) throw new Error("file is required for GroqAi.transcribe");

    let input;
    if (typeof file === "string") {
      // path
      const fs = require("fs");
      input = fs.createReadStream(file);
    } else {
      // buffer or stream
      input = file;
    }

    const models = model ? [model] : this.models.length ? this.models : [
      "whisper-large-v3-turbo",
    ];

    const runOnce = async (targetModel) => {
      const work = this.client.audio.transcriptions.create({
        file: input,
        model: targetModel,
        temperature,
        response_format: "verbose_json",
      });

      const resp =
        this.requestTimeoutMs > 0
          ? await Promise.race([
              work,
              new Promise((_, reject) =>
                setTimeout(
                  () =>
                    reject(
                      new Error(
                        `Groq transcription model ${targetModel} timed out after ${this.requestTimeoutMs}ms`,
                      ),
                    ),
                  this.requestTimeoutMs,
                ),
              ),
            ])
          : await work;

      const text = resp?.text || "";
      if (!text) throw new Error(`Groq transcription returned empty text for ${targetModel}`);
      return text.trim();
    };

    if (!this.firstToFinish || models.length === 1) {
      let lastError;
      for (const m of models) {
        try {
          const text = await runOnce(m);
          this.lastUsedModel = m;
          return text;
        } catch (err) {
          lastError = err;
          console.error("[GroqAI] transcription failed:", err?.message || err);
        }
      }
      throw lastError || new Error("All Groq transcription models failed");
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      let remaining = models.length;
      let lastError;

      for (const m of models) {
        (async () => {
          try {
            const text = await runOnce(m);
            if (!settled) {
              settled = true;
              this.lastUsedModel = m;
              resolve(text);
            }
          } catch (err) {
            lastError = err;
            console.error("[GroqAI] transcription failed:", err?.message || err);
          } finally {
            remaining -= 1;
            if (!settled && remaining === 0) {
              settled = true;
              reject(lastError || new Error("All Groq transcription models failed"));
            }
          }
        })();
      }
    });
  }
}

/**
 * Groq version of AiWithHistory
 * Same API: ask(chatId, {...}), clear()
 */
class GroqAiWithHistory extends GroqAi {
   constructor({
     memoryStore,
     memoryScope = "default",
     historyLimit = 10,
     ...options
   } = {}) {
     if (!memoryStore) {
       throw new Error("memoryStore is required for GroqAiWithHistory");
     }
     super(options);
     this.memoryStore = memoryStore;
     this.memoryScope = memoryScope;
     this.historyLimit = historyLimit;
   }

  formatStoredContent(content) {
    if (Array.isArray(content)) return content.join("\n");
    if (typeof content === "object" && content !== null) {
      return JSON.stringify(content);
    }
    return content;
  }

  async ask(chatId, { system, user, attachments = [] } = {}) {
    if (!chatId) throw new Error("chatId is required for GroqAiWithHistory");

    const history = await this.memoryStore.getHistory(
      chatId,
      this.memoryScope,
      this.historyLimit,
    );

    const formattedHistory = history.map((entry) => ({
      role: entry.role === "assistant" ? "assistant" : "user",
      content: this.formatStoredContent(entry.content),
    }));

    const { contentForHistory } = this.buildUserMessagePayload({
      user,
      attachments,
    });

    const response = await super.ask({
      system,
      user,
      attachments,
      messages: formattedHistory,
    });

    const toPersist = [];
    if (contentForHistory !== null && contentForHistory !== undefined) {
      toPersist.push({ role: "user", content: contentForHistory });
    }
    if (response) {
      toPersist.push({ role: "assistant", content: response });
    }

    if (toPersist.length) {
      this.memoryStore
        .appendMessages(chatId, this.memoryScope, toPersist)
        .catch((e) => {
          console.error("[Groq Memory] Failed to persist:", e);
        });
    }

    return response;
  }

  async clear(chatId) {
    await this.memoryStore.clearHistory(chatId, this.memoryScope);
  }
}

/**
 * Mistral version of the Ai class.
 * Same constructor shape, same ask() behavior, fallbacks included.
 */
class MistralAi {
  /**
   * Sets up the Mistral helper with model preferences and request defaults.
   * @param {{
   *  apiKey: string,
   *  model?: string,
   *  fallbackModels?: string[],
   *  temperature?: number,
   *  maxTokens?: number,
   *  requestTimeoutMs?: number,
   *  firstToFinish?: boolean
   * }} [options]
   */
  constructor({
     apiKey,
     model = "mistral-small-latest",
     fallbackModels = [],
     temperature = 0.7,
     maxTokens = 1000,
     requestTimeoutMs = 20000,
     firstToFinish = false,
   } = {}) {
     if (!apiKey) {
       throw new Error("apiKey is required for MistralAi");
     }
     this.models = [model, ...fallbackModels].filter(Boolean);
     this.temperature = temperature;
     this.maxTokens = maxTokens;
     this.requestTimeoutMs = Number(requestTimeoutMs) || 0;
     this.client = new Mistral({ apiKey });
     this.lastUsedModel = null;
     this.firstToFinish = firstToFinish;
   }

  /**
   * Converts mixed message content into a consistent array so we can tack on attachments.
   * @param {string|Array|undefined} content
   * @returns {Array}
   */
  ensureContentArray(content) {
    if (Array.isArray(content)) {
      return [...content];
    }
    if (typeof content === "string" && content.length) {
      return [{ type: "text", text: content }];
    }
    return [];
  }

  /**
   * Turns an attachment descriptor into a Mistral-friendly multimodal chunk.
   * @param {AiAttachment} attachment
   * @returns {Record<string, any>|null}
   */
  normalizeAttachment(attachment) {
    if (!attachment || !attachment.type) return null;
    const type = String(attachment.type).toLowerCase();
    const { url, data, mimeType, format } = attachment;

    if (type === "image") {
      if (url) {
        return { type: "image_url", imageUrl: url };
      }

      const dataUri = this.toDataUri(data, mimeType || "image/png");
      if (!dataUri) return null;
      return { type: "image_url", imageUrl: dataUri };
    }

    // Mistral supports some video formats through image conversion
    if (type === "video") {
      console.warn(
        "[MistralAI] Video attachments should be converted to images for Mistral",
      );
      return null;
    }

    return null;
  }

  /**
   * Converts incoming blobs into base64 so the API can ingest them inline.
   * @param {string|Buffer|undefined} data
   * @returns {string|null}
   */
  toBase64(data) {
    if (!data) return null;
    if (Buffer.isBuffer(data)) {
      return data.toString("base64");
    }
    if (typeof data === "string") {
      const trimmed = data.trim();
      if (!trimmed) return null;
      if (trimmed.startsWith("data:")) {
        const base64Part = trimmed.substring(trimmed.indexOf(",") + 1);
        return base64Part ? base64Part.trim() : null;
      }
      const base64Like = trimmed.replace(/\s+/g, "");
      if (/^[A-Za-z0-9+/]+={0,2}$/.test(base64Like)) {
        return base64Like;
      }
      return Buffer.from(trimmed).toString("base64");
    }
    return null;
  }

  /**
   * Shapes data blobs into data URIs when the API expects them that way (mostly for images).
   * @param {string|Buffer|undefined} data
   * @param {string} mimeType
   * @returns {string|null}
   */
  toDataUri(data, mimeType) {
    if (!data) return null;
    if (typeof data === "string" && data.trim().startsWith("data:")) {
      return data.trim();
    }
    const encoded = this.toBase64(data);
    if (!encoded) return null;
    return `data:${mimeType};base64,${encoded}`;
  }

  /**
   * Builds the outgoing user message with attachments for Mistral.
   * @param {{ user?: any, attachments?: AiAttachment[] }} [options]
   * @returns {{ content: string|Array, contentForHistory: string|Array|null }}
   */
  buildUserMessagePayload({ user, attachments = [] } = {}) {
    const normalizedAttachments = Array.isArray(attachments)
      ? attachments
          .map((attachment) => this.normalizeAttachment(attachment))
          .filter(Boolean)
      : [];

    const baseContent = this.ensureContentArray(
      typeof user === "string" ? user : user?.content,
    );

    const contentPieces = [...baseContent, ...normalizedAttachments];

    if (!contentPieces.length) {
      return { content: "", contentForHistory: null };
    }

    // For Mistral, if we have attachments, we need to use the content array format
    // Otherwise, we can use a simple string
    const finalContent =
      normalizedAttachments.length > 0
        ? contentPieces
        : typeof user === "string"
          ? user
          : user?.content || "";

    return {
      content: finalContent,
      contentForHistory:
        normalizedAttachments.length > 0 ? contentPieces : finalContent,
    };
  }

  extractText(resp) {
    try {
      return resp.choices?.[0]?.message?.content?.trim() || "";
    } catch {
      return "";
    }
  }

  async ask({ system, user, messages = [], attachments = [] } = {}) {
    if (!this.models.length) throw new Error("No Mistral models configured");

    const mistralMessages = [];
    if (system) mistralMessages.push({ role: "system", content: system });

    if (messages.length) {
      mistralMessages.push(
        ...messages.map((m) => {
          // Handle plain objects that already have role/content
          if (m.role && m.content) {
            return {
              role: m.role,
              content:
                typeof m.content === "string"
                  ? m.content
                  : JSON.stringify(m.content),
            };
          }
          // Handle LangChain message objects
          if (typeof m._getType === "function") {
            return {
              role: m._getType() === "ai" ? "assistant" : "user",
              content:
                typeof m.content === "string"
                  ? m.content
                  : JSON.stringify(m.content),
            };
          }
          // Fallback for unknown format
          return {
            role: "user",
            content: JSON.stringify(m),
          };
        }),
      );
    }

    // Handle user message with attachments
    if (user || attachments.length > 0) {
      const { content } = this.buildUserMessagePayload({ user, attachments });
      mistralMessages.push({ role: "user", content });
    }

    if (!this.firstToFinish || this.models.length === 1) {
      let lastError;
      for (const model of this.models) {
        try {
          const work = this.client.chat.complete({
            model,
            messages: mistralMessages,
            temperature: this.temperature,
            maxTokens: this.maxTokens,
          });

          const resp =
            this.requestTimeoutMs > 0
              ? await Promise.race([
                  work,
                  new Promise((_, reject) =>
                    setTimeout(
                      () =>
                        reject(
                          new Error(
                            `Model ${model} timed out after ${this.requestTimeoutMs}ms`,
                          ),
                        ),
                      this.requestTimeoutMs,
                    ),
                  ),
                ])
              : await work;

          const text = this.extractText(resp);
          if (!text) throw new Error(`Empty response from ${model}`);

          this.lastUsedModel = model;
          return text;
        } catch (err) {
          lastError = err;
          console.error(`[MistralAI] ${model} failed:`, err?.message || err);
        }
      }

      throw lastError || new Error("All Mistral models failed");
    }

    return new Promise((resolve, reject) => {
      let settled = false;
      let remaining = this.models.length;
      let lastError;

      for (const model of this.models) {
        (async () => {
          try {
            const work = this.client.chat.complete({
              model,
              messages: mistralMessages,
              temperature: this.temperature,
              maxTokens: this.maxTokens,
            });

            const resp =
              this.requestTimeoutMs > 0
                ? await Promise.race([
                    work,
                    new Promise((_, reject) =>
                      setTimeout(
                        () =>
                          reject(
                            new Error(
                              `Model ${model} timed out after ${this.requestTimeoutMs}ms`,
                            ),
                          ),
                        this.requestTimeoutMs,
                      ),
                    ),
                  ])
                : await work;

            const text = this.extractText(resp);
            if (!text) throw new Error(`Empty response from ${model}`);

            if (!settled) {
              settled = true;
              this.lastUsedModel = model;
              resolve(text);
            }
          } catch (err) {
            lastError = err;
            console.error(`[MistralAI] ${model} failed:`, err?.message || err);
          } finally {
            remaining -= 1;
            if (!settled && remaining === 0) {
              settled = true;
              reject(lastError || new Error("All Mistral models failed"));
            }
          }
        })();
      }
    });
  }

  /**
   * Transcribes an audio file using Mistral Voxtral.
   * @param {Object} options
   * @param {string|Buffer|ReadableStream} options.file - Path, URL, buffer, or stream
   * @param {string} [options.model="voxtral-mini-latest"]
   * @param {string} [options.language]
   * @param {Array<string>} [options.timestamp_granularities] - Timestamp granularities for transcription
   * @returns {Promise<string>}
   */
  async transcribe({
    file,
    model = "voxtral-mini-latest",
    language,
    timestamp_granularities,
  } = {}) {
    if (!file) throw new Error("file is required for MistralAi.transcribe");

    const params = { model };
    if (language) params.language = language;
    if (timestamp_granularities) params.timestamp_granularities = timestamp_granularities;

    let inputFile = file;
    if (file && typeof file.pipe === 'function') {
      // it's a stream, read to buffer
      inputFile = await new Promise((resolve, reject) => {
        const chunks = [];
        file.on('data', chunk => chunks.push(chunk));
        file.on('end', () => resolve(Buffer.concat(chunks)));
        file.on('error', reject);
      });
    }

    if (typeof inputFile === "string") {
      if (inputFile.startsWith("http")) {
        params.fileUrl = inputFile;
      } else {
        // path
        const fs = require("fs");
        const path = require("path");
        const content = fs.readFileSync(inputFile);
        const fileName = path.basename(inputFile);
        params.file = { fileName, content };
      }
    } else if (Buffer.isBuffer(inputFile)) {
      // buffer
      params.file = { fileName: "audio.mp3", content: inputFile };
    } else {
      throw new Error("file must be a string (path or URL), Buffer, or ReadableStream");
    }

    try {
      const models = model ? [model] : this.models.length ? this.models : [
        "voxtral-mini-latest",
      ];

      const runOnce = async (targetModel) => {
        const work = this.client.audio.transcriptions.complete({
          ...params,
          model: targetModel,
        });

        const resp =
          this.requestTimeoutMs > 0
            ? await Promise.race([
                work,
                new Promise((_, reject) =>
                  setTimeout(
                    () =>
                      reject(
                        new Error(
                          `Mistral transcription model ${targetModel} timed out after ${this.requestTimeoutMs}ms`,
                        ),
                      ),
                    this.requestTimeoutMs,
                  ),
                ),
              ])
            : await work;

        const text = resp?.text || "";
        if (!text) throw new Error(`Mistral transcription returned empty text for ${targetModel}`);
        return text.trim();
      };

      if (!this.firstToFinish || models.length === 1) {
        let lastError;
        for (const m of models) {
          try {
            const text = await runOnce(m);
            this.lastUsedModel = m;
            return text;
          } catch (err) {
            lastError = err;
            console.error("[MistralAI] transcription failed:", err?.message || err);
          }
        }
        throw lastError || new Error("All Mistral transcription models failed");
      }

      return await new Promise((resolve, reject) => {
        let settled = false;
        let remaining = models.length;
        let lastError;

        for (const m of models) {
          (async () => {
            try {
              const text = await runOnce(m);
              if (!settled) {
                settled = true;
                this.lastUsedModel = m;
                resolve(text);
              }
            } catch (err) {
              lastError = err;
              console.error("[MistralAI] transcription failed:", err?.message || err);
            } finally {
              remaining -= 1;
              if (!settled && remaining === 0) {
                settled = true;
                reject(
                  lastError || new Error("All Mistral transcription models failed"),
                );
              }
            }
          })();
        }
      });
    } catch (err) {
      console.error("[MistralAI] transcription failed:", err?.message || err);
      throw err;
    }
  }

  async classify(
    inputs,
    { model = "mistral-moderation-latest", requestTimeoutMs } = {},
  ) {
    if (inputs == null || (Array.isArray(inputs) && inputs.length === 0)) {
      throw new Error("inputs is required for classify()");
    }

    const isArrayInput = Array.isArray(inputs);
    const normalizedInputs = isArrayInput ? inputs : [inputs];

    const timeout =
      requestTimeoutMs !== undefined
        ? Number(requestTimeoutMs)
        : Number(this.requestTimeoutMs) || 0;

    const work = this.client.classifiers.moderate({
      model,
      inputs: normalizedInputs,
    });

    const resp =
      timeout > 0
        ? await Promise.race([
            work,
            new Promise((_, reject) =>
              setTimeout(
                () =>
                  reject(
                  new Error(
                    `Moderation model ${model} timed out after ${timeout}ms`,
                  ),
                ),
                timeout,
              ),
            ),
          ])
        : await work;

    const mapped = (resp.results || []).map((r) => ({
      categories: r.categories || {},
      scores: r.category_scores || {},
    }));

    // Single input → first item, multi-input → whole array
    return isArrayInput ? mapped : mapped[0] || { categories: {}, scores: {} };
  }
}

/**
 * Mistral version of AiWithHistory
 * Same API: ask(chatId, {...}), clear()
 */
class MistralAiWithHistory extends MistralAi {
   constructor({
     memoryStore,
     memoryScope = "default",
     historyLimit = 10,
     ...options
   } = {}) {
     if (!memoryStore) {
       throw new Error("memoryStore is required for MistralAiWithHistory");
     }
     super(options);
     this.memoryStore = memoryStore;
     this.memoryScope = memoryScope;
     this.historyLimit = historyLimit;
   }

  formatStoredContent(content) {
    if (Array.isArray(content)) return content.join("\n");
    if (typeof content === "object" && content !== null) {
      return JSON.stringify(content);
    }
    return content;
  }

  async ask(chatId, { system, user, attachments = [] } = {}) {
    if (!chatId) throw new Error("chatId is required for MistralAiWithHistory");

    const history = await this.memoryStore.getHistory(
      chatId,
      this.memoryScope,
      this.historyLimit,
    );

    const formattedHistory = history.map((entry) => ({
      role: entry.role === "assistant" ? "assistant" : "user",
      content: this.formatStoredContent(entry.content),
    }));

    const { contentForHistory } = this.buildUserMessagePayload({
      user,
      attachments,
    });

    const response = await super.ask({
      system,
      user,
      attachments,
      messages: formattedHistory,
    });

    const toPersist = [];
    if (contentForHistory !== null && contentForHistory !== undefined) {
      toPersist.push({ role: "user", content: contentForHistory });
    }
    if (response) {
      toPersist.push({ role: "assistant", content: response });
    }

    if (toPersist.length) {
      this.memoryStore
        .appendMessages(chatId, this.memoryScope, toPersist)
        .catch((e) => {
          console.error("[Mistral Memory] Failed to persist:", e);
        });
    }

    return response;
  }

  async clear(chatId) {
    await this.memoryStore.clearHistory(chatId, this.memoryScope);
  }
}

/**
 * High-level helper that can talk to multiple underlying providers (OpenRouter, Groq, Mistral)
 * using a single, unified API.
 *
 * It mirrors the behavior of the provider-specific helpers (Ai, GroqAi, MistralAi) while
 * supporting cross-provider fallbacks and optional first-to-finish racing.
 */
class MultiProviderAi {
  constructor({
     apiKeys = {
      openrouter: "",
      mistral: "",
      groq: "",
     },
     model = {
      provider: "mistral",
      name: "mistral-small-latest"
     },
     fallbackModels = {
      openrouter: [],
      mistral: [],
      groq: []
     },
     temperature = 0.7,
     maxTokens = 1000,
     requestTimeoutMs = 20000,
     firstToFinish = false,
   } = {}) {
     if (!apiKeys || typeof apiKeys !== "object") {
       throw new Error("apiKeys must be a non-null object with provider keys");
     }
     this.temperature = temperature;
     this.maxTokens = maxTokens;
     this.requestTimeoutMs = Number(requestTimeoutMs) || 0;
     this.firstToFinish = firstToFinish;

     this.primaryProvider = model && typeof model === "object" ? model.provider : undefined;

     this.clients = {};

     const providerConfigs = {
       openrouter: {
         classRef: Ai,
         modelKey: "openrouter",
       },
       mistral: {
         classRef: MistralAi,
         modelKey: "mistral",
       },
       groq: {
         classRef: GroqAi,
         modelKey: "groq",
       },
     };

     for (const [provider, key] of Object.entries(apiKeys)) {
       if (!key) continue;
       const cfg = providerConfigs[provider];
       if (!cfg) continue;

       const isPrimary =
         model &&
         typeof model === "object" &&
         model.provider === provider &&
         typeof model.name === "string" &&
         model.name.length > 0;

       const ctorOptions = {
         apiKey: key,
         temperature: this.temperature,
         maxTokens: this.maxTokens,
         requestTimeoutMs: this.requestTimeoutMs,
         firstToFinish: this.firstToFinish,
       };

       if (isPrimary) {
         ctorOptions.model = model.name;
       }

       const fallbacks = fallbackModels?.[provider];
       if (Array.isArray(fallbacks) && fallbacks.length) {
         ctorOptions.fallbackModels = fallbacks;
       }

       this.clients[provider] = new cfg.classRef(ctorOptions);
     }

     this.lastUsedModel = null;
   }

   getOrderedProviders() {
     const available = Object.keys(this.clients);
     if (!available.length) return [];

     const preferred = this.primaryProvider && available.includes(this.primaryProvider)
       ? this.primaryProvider
       : available[0];

     const rest = available.filter((p) => p !== preferred);
     return [preferred, ...rest];
   }

   async ask({ system, user, messages = [], attachments = [] } = {}) {
     const providers = this.getOrderedProviders();
     if (!providers.length) {
       throw new Error("No AI providers configured for MultiProviderAi");
     }

     if (!this.firstToFinish || providers.length === 1) {
       let lastError;
       for (const provider of providers) {
         const client = this.clients[provider];
         if (!client || typeof client.ask !== "function") continue;
         try {
           const resp = await client.ask({ system, user, messages, attachments });
           this.lastUsedModel = { provider, model: client.lastUsedModel || null };
           return resp;
         } catch (err) {
           lastError = err;
           console.error(`[MultiProviderAI] ${provider} failed:`, err?.message || err);
         }
       }
       throw lastError || new Error("All AI providers failed");
     }

     return new Promise((resolve, reject) => {
       let settled = false;
       let remaining = providers.length;
       let lastError;

       for (const provider of providers) {
         const client = this.clients[provider];
         if (!client || typeof client.ask !== "function") {
           remaining -= 1;
           if (!settled && remaining === 0) {
             reject(lastError || new Error("All AI providers failed"));
           }
           continue;
         }

         (async () => {
           try {
             const resp = await client.ask({ system, user, messages, attachments });
             if (!settled) {
               settled = true;
               this.lastUsedModel = { provider, model: client.lastUsedModel || null };
               resolve(resp);
             }
           } catch (err) {
             lastError = err;
             console.error(`[MultiProviderAI] ${provider} failed:`, err?.message || err);
           } finally {
             remaining -= 1;
             if (!settled && remaining === 0) {
               settled = true;
               reject(lastError || new Error("All AI providers failed"));
             }
           }
         })();
       }
     });
   }

   async transcribe(options = {}) {
     const providers = this.getOrderedProviders().filter((p) =>
       typeof this.clients[p]?.transcribe === "function",
     );

     if (!providers.length) {
       throw new Error("No providers with transcribe() configured for MultiProviderAi");
     }

     if (!this.firstToFinish || providers.length === 1) {
       let lastError;
       for (const provider of providers) {
         const client = this.clients[provider];
         try {
           const text = await client.transcribe(options);
           this.lastUsedModel = { provider, model: client.lastUsedModel || null };
           return text;
         } catch (err) {
           lastError = err;
           console.error(
             `[MultiProviderAI] transcription via ${provider} failed:`,
             err?.message || err,
           );
         }
       }
       throw lastError || new Error("All transcription providers failed");
     }

     return new Promise((resolve, reject) => {
       let settled = false;
       let remaining = providers.length;
       let lastError;

       for (const provider of providers) {
         const client = this.clients[provider];
         (async () => {
           try {
             const text = await client.transcribe(options);
             if (!settled) {
               settled = true;
               this.lastUsedModel = { provider, model: client.lastUsedModel || null };
               resolve(text);
             }
           } catch (err) {
             lastError = err;
             console.error(
               `[MultiProviderAI] transcription via ${provider} failed:`,
               err?.message || err,
             );
           } finally {
             remaining -= 1;
             if (!settled && remaining === 0) {
               settled = true;
               reject(lastError || new Error("All transcription providers failed"));
             }
           }
         })();
       }
     });
   }

   async classify(inputs, options = {}) {
     if (inputs == null || (Array.isArray(inputs) && inputs.length === 0)) {
       throw new Error("inputs is required for classify()");
     }

     const providers = this.getOrderedProviders().filter((p) =>
       typeof this.clients[p]?.classify === "function",
     );

     if (!providers.length) {
       throw new Error("No providers with classify() configured for MultiProviderAi");
     }

     const provider = providers[0];
     const client = this.clients[provider];
     const result = await client.classify(inputs, options);
     this.lastUsedModel = { provider, model: client.lastUsedModel || null };
     return result;
   }
 }

/**
 * Multi-provider AI helper that also persists and reuses per-chat history via AiMemoryStore.
 *
 * This behaves like AiWithHistory/GroqAiWithHistory/MistralAiWithHistory but delegates the
 * actual completion work to MultiProviderAi so you can seamlessly span providers.
 */
class MultiProviderAiWithHistory extends MultiProviderAi {
  formatStoredContent(content) {
    if (Array.isArray(content)) {
      return { content };
    }
    if (content && typeof content === "object" && "content" in content) {
      return content;
    }
    return content;
  }

  constructor({
    memoryStore,
    memoryScope = "default",
    historyLimit = 10,
    ...options
  } = {}) {
    if (!memoryStore) {
      throw new Error("memoryStore is required for MultiProviderAiWithHistory");
    }
    super(options);
    this.memoryStore = memoryStore;
    this.memoryScope = memoryScope;
    this.historyLimit = historyLimit;
  }

  ensureContentArray(content) {
    if (Array.isArray(content)) {
      return [...content];
    }
    if (typeof content === "string" && content.length) {
      return [{ type: "text", text: content }];
    }
    return [];
  }

  toBase64(data) {
    if (!data) return null;
    if (Buffer.isBuffer(data)) {
      return data.toString("base64");
    }
    if (typeof data === "string") {
      const trimmed = data.trim();
      if (!trimmed) return null;
      if (trimmed.startsWith("data:")) {
        const base64Part = trimmed.substring(trimmed.indexOf(",") + 1);
        return base64Part ? base64Part.trim() : null;
      }
      const base64Like = trimmed.replace(/\s+/g, "");
      if (/^[A-Za-z0-9+/]+={0,2}$/.test(base64Like)) {
        return base64Like;
      }
      return Buffer.from(trimmed).toString("base64");
    }
    return null;
  }

  toDataUri(data, mimeType) {
    if (!data) return null;
    if (typeof data === "string" && data.trim().startsWith("data:")) {
      return data.trim();
    }
    const encoded = this.toBase64(data);
    if (!encoded) return null;
    return `data:${mimeType};base64,${encoded}`;
  }

  mimeTypeToFormat(mimeType) {
    if (!mimeType) return undefined;
    const [, subtype] = mimeType.split("/");
    if (!subtype) return undefined;
    return subtype.split(";")[0] || undefined;
  }

  normalizeAttachment(attachment) {
    if (!attachment || !attachment.type) return null;
    const type = String(attachment.type).toLowerCase();
    const { url, data, mimeType, format } = attachment;

    if (type === "image") {
      if (url) {
        return { type: "image_url", image_url: { url } };
      }

      const dataUri = this.toDataUri(data, mimeType || "image/png");
      if (!dataUri) return null;
      return { type: "image_url", image_url: { url: dataUri } };
    }

    if (type === "video") {
      if (url) {
        return { type: "input_video", video: { url } };
      }

      const encoded = this.toBase64(data);
      if (!encoded) return null;
      const payload = { data: encoded };
      const resolvedFormat =
        format || (mimeType ? this.mimeTypeToFormat(mimeType) : undefined);
      if (resolvedFormat) {
        payload.format = resolvedFormat;
      }
      return { type: "input_video", video: payload };
    }

    return null;
  }

  buildUserMessagePayload({ user, attachments = [] } = {}) {
    const normalizedAttachments = Array.isArray(attachments)
      ? attachments
          .map((attachment) => this.normalizeAttachment(attachment))
          .filter(Boolean)
      : [];

    if (user instanceof HumanMessage) {
      if (!normalizedAttachments.length) {
        return { message: user, contentForHistory: user.content };
      }
      const contentPieces = this.ensureContentArray(user.content).concat(
        normalizedAttachments,
      );
      const rebuilt = new HumanMessage({
        content: contentPieces,
        additional_kwargs: user.additional_kwargs,
        name: user.name,
      });
      return { message: rebuilt, contentForHistory: contentPieces };
    }

    if (Array.isArray(user)) {
      const contentPieces = [...user, ...normalizedAttachments];
      if (!contentPieces.length) {
        return { message: null, contentForHistory: null };
      }
      return {
        message: new HumanMessage({ content: contentPieces }),
        contentForHistory: contentPieces,
      };
    }

    if (
      user &&
      typeof user === "object" &&
      user !== null &&
      "content" in user
    ) {
      const baseContent = this.ensureContentArray(user.content).concat(
        normalizedAttachments,
      );
      if (!baseContent.length) {
        return { message: null, contentForHistory: null };
      }
      return {
        message: new HumanMessage({
          ...user,
          content: baseContent,
        }),
        contentForHistory: baseContent,
      };
    }

    const trimmedUser = typeof user === "string" ? user : undefined;

    if (normalizedAttachments.length === 0) {
      if (typeof trimmedUser === "string" && trimmedUser.length) {
        return {
          message: new HumanMessage(trimmedUser),
          contentForHistory: trimmedUser,
        };
      }
      return { message: null, contentForHistory: null };
    }

    const contentPieces = [];
    if (typeof trimmedUser === "string" && trimmedUser.length) {
      contentPieces.push({ type: "text", text: trimmedUser });
    }
    contentPieces.push(...normalizedAttachments);

    if (!contentPieces.length) {
      return { message: null, contentForHistory: null };
    }

    return {
      message: new HumanMessage({ content: contentPieces }),
      contentForHistory: contentPieces,
    };
  }

  async ask(chatId, { system, user, attachments = [] } = {}) {
    if (!chatId) {
      throw new Error("chatId is required for MultiProviderAiWithHistory");
    }

    const historyEntries = await this.memoryStore.getHistory(
      chatId,
      this.memoryScope,
      this.historyLimit,
    );

    const formattedHistory = historyEntries.map((entry) => {
      const payload = this.formatStoredContent(entry.content);
      return entry.role === "assistant"
        ? new AIMessage(payload)
        : new HumanMessage(payload);
    });

    const { contentForHistory } = this.buildUserMessagePayload({
      user,
      attachments,
    });

    const response = await super.ask({
      system,
      user,
      attachments,
      messages: formattedHistory,
    });

    const toPersist = [];
    if (contentForHistory !== null && contentForHistory !== undefined) {
      toPersist.push({ role: "user", content: contentForHistory });
    }
    if (response) {
      toPersist.push({ role: "assistant", content: response });
    }

    if (toPersist.length) {
      this.memoryStore
        .appendMessages(chatId, this.memoryScope, toPersist)
        .catch((err) => {
          console.error("[MultiProvider Memory] Failed to persist chat:", err);
        });
    }

    return response;
  }

  async clear(chatId) {
    await this.memoryStore.clearHistory(chatId, this.memoryScope);
  }
}

module.exports = {
  Ai,
  AiWithHistory,
  AiMemoryStore,
  GroqAi,
  GroqAiWithHistory,
  MistralAi,
  MistralAiWithHistory,
  MultiProviderAi,
  MultiProviderAiWithHistory,
};
