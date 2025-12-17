"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Trash2, Plus, Scale, BookOpen, AlertCircle, Loader2, MessageSquare, ExternalLink } from 'lucide-react';

const LegalAssistChat = () => {
  const [threads, setThreads] = useState([]);
  const [currentThreadId, setCurrentThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [followUp, setFollowUp] = useState('');
  const [routingInfo, setRoutingInfo] = useState('');
  const messagesEndRef = useRef(null);
  const [streamingMessage, setStreamingMessage] = useState('');

  const API_BASE = 'http://localhost:8000';

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingMessage]);

  // Load threads on mount
  useEffect(() => {
    loadThreads();
  }, []);

  const loadThreads = async () => {
    try {
      const res = await fetch(`${API_BASE}/threads`);
      const data = await res.json();
      setThreads(data);
    } catch (err) {
      console.error('Failed to load threads:', err);
    }
  };

  const createNewThread = () => {
    const newThreadId = `thread_${Date.now()}`;
    setCurrentThreadId(newThreadId);
    setMessages([]);
    setFollowUp('');
    setRoutingInfo('');
  };

  const loadThread = async (threadId) => {
    try {
      const res = await fetch(`${API_BASE}/history/${threadId}`);
      const data = await res.json();
      setCurrentThreadId(threadId);
      setMessages(data.messages || []);
      setFollowUp('');
      setRoutingInfo('');
    } catch (err) {
      console.error('Failed to load thread:', err);
    }
  };

  const deleteThread = async (threadId, e) => {
    e.stopPropagation();
    try {
      await fetch(`${API_BASE}/threads/${threadId}`, { method: 'DELETE' });
      loadThreads();
      if (currentThreadId === threadId) {
        setCurrentThreadId(null);
        setMessages([]);
      }
    } catch (err) {
      console.error('Failed to delete thread:', err);
    }
  };

  const sendMessage = async (text = input) => {
    if (!text.trim() || isLoading) return;
    if (!currentThreadId) createNewThread();

    const threadId = currentThreadId || `thread_${Date.now()}`;
    if (!currentThreadId) setCurrentThreadId(threadId);

    const userMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingMessage('');
    setFollowUp('');
    setRoutingInfo('');

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, thread_id: threadId })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'token') {
                accumulatedText += data.content;
                setStreamingMessage(accumulatedText);
              } else if (data.type === 'done') {
                setMessages(prev => [...prev, { role: 'assistant', content: accumulatedText }]);
                setStreamingMessage('');
              } else if (data.type === 'follow_up') {
                setFollowUp(data.content);
              } else if (data.type === 'routing') {
                setRoutingInfo(data.content);
              } else if (data.type === 'citations') {
                // Citations could be displayed separately if needed
              } else if (data.type === 'error') {
                console.error('Stream error:', data.content);
              }
            } catch (err) {
              // Ignore JSON parse errors for incomplete chunks
            }
          }
        }
      }

      loadThreads();
    } catch (err) {
      console.error('Failed to send message:', err);
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '❌ Sorry, I encountered an error. Please try again.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const useFollowUp = () => {
    if (followUp) {
      setInput(followUp);
      setFollowUp('');
    }
  };

  return (
    <div className="flex h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Sidebar */}
      <div className="w-80 bg-slate-800/50 backdrop-blur-xl border-r border-slate-700/50 flex flex-col">
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center gap-3 mb-4">
            <Scale className="w-8 h-8 text-amber-400" />
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-transparent">
                Legal Assist
              </h1>
              <p className="text-xs text-slate-400">Indian Legal RAG System</p>
            </div>
          </div>
          <button
            onClick={createNewThread}
            className="w-full bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 text-white py-3 rounded-lg flex items-center justify-center gap-2 font-medium transition-all shadow-lg hover:shadow-xl"
          >
            <Plus className="w-5 h-5" />
            New Legal Query
          </button>
        </div>

        {/* Thread List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          {threads.length === 0 ? (
            <div className="text-center text-slate-500 py-8">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">No conversations yet</p>
            </div>
          ) : (
            threads.map((thread) => (
              <div
                key={thread.thread_id}
                onClick={() => loadThread(thread.thread_id)}
                className={`p-3 rounded-lg cursor-pointer transition-all group ${
                  currentThreadId === thread.thread_id
                    ? 'bg-gradient-to-r from-amber-500/20 to-orange-500/20 border border-amber-500/30'
                    : 'bg-slate-700/30 hover:bg-slate-700/50 border border-transparent'
                }`}
              >
                <div className="flex justify-between items-start gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate text-white">
                      {thread.preview || 'New conversation'}
                    </p>
                    <p className="text-xs text-slate-400 mt-1">
                      {thread.message_count} messages
                    </p>
                  </div>
                  <button
                    onClick={(e) => deleteThread(thread.thread_id, e)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1.5 hover:bg-red-500/20 rounded text-red-400"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Footer Info */}
        <div className="p-4 border-t border-slate-700/50 text-xs text-slate-400">
          <div className="flex items-center gap-2 mb-2">
            <BookOpen className="w-4 h-4" />
            <span className="font-medium">Supported Domains:</span>
          </div>
          <div className="space-y-1 pl-6">
            <div>• Criminal Law (IPC, CrPC)</div>
            <div>• Civil & Family Law</div>
            <div>• Property & Labour Law</div>
            <div>• Constitutional Law</div>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-slate-800/50 backdrop-blur-xl border-b border-slate-700/50 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center">
                  <Scale className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h2 className="font-semibold text-lg">Legal Assistant</h2>
                  <p className="text-sm text-slate-400">Ask any legal question about Indian law</p>
                </div>
              </div>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 bg-slate-700/50 hover:bg-slate-700 rounded-lg text-sm transition-colors"
              >
                <ExternalLink className="w-4 h-4" />
                API Docs
              </a>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 0 && !streamingMessage && (
              <div className="text-center py-12">
                <div className="w-20 h-20 bg-gradient-to-br from-amber-400/20 to-orange-500/20 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Scale className="w-10 h-10 text-amber-400" />
                </div>
                <h3 className="text-2xl font-bold mb-3 bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-transparent">
                  Welcome to Legal Assist
                </h3>
                <p className="text-slate-400 mb-8 max-w-md mx-auto">
                  Your AI-powered Indian legal assistant. Ask questions about IPC, CrPC, family law, property disputes, and more.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
                  {[
                    'What is Section 420 IPC?',
                    'How to file for divorce?',
                    'Property inheritance rights',
                    'Employee termination laws'
                  ].map((example, i) => (
                    <button
                      key={i}
                      onClick={() => setInput(example)}
                      className="p-4 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-amber-500/30 rounded-lg text-left transition-all group"
                    >
                      <p className="text-sm text-slate-300 group-hover:text-white">{example}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                {msg.role === 'assistant' && (
                  <div className="w-8 h-8 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center flex-shrink-0">
                    <Scale className="w-5 h-5 text-white" />
                  </div>
                )}
                <div className={`max-w-3xl ${msg.role === 'user' ? 'order-first' : ''}`}>
                  <div
                    className={`rounded-2xl p-4 ${
                      msg.role === 'user'
                        ? 'bg-gradient-to-r from-amber-500 to-orange-500 text-white'
                        : 'bg-slate-800/70 backdrop-blur-sm border border-slate-700/50'
                    }`}
                  >
                    <div className="prose prose-invert max-w-none">
                      <ReactMarkdown content={msg.content} />
                    </div>
                  </div>
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 bg-slate-700 rounded-full flex items-center justify-center flex-shrink-0">
                    <span className="text-sm font-medium">You</span>
                  </div>
                )}
              </div>
            ))}

            {streamingMessage && (
              <div className="flex gap-4">
                <div className="w-8 h-8 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <Scale className="w-5 h-5 text-white" />
                </div>
                <div className="max-w-3xl">
                  <div className="rounded-2xl p-4 bg-slate-800/70 backdrop-blur-sm border border-slate-700/50">
                    <div className="prose prose-invert max-w-none">
                      <ReactMarkdown content={streamingMessage} />
                    </div>
                    <div className="flex items-center gap-2 mt-2 text-amber-400">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-xs">Generating response...</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {routingInfo && (
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-3 text-sm">
                <div className="flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="font-medium text-blue-300 mb-1">Query Analysis</p>
                    <p className="text-blue-200/80">{routingInfo}</p>
                  </div>
                </div>
              </div>
            )}

            {followUp && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                <p className="text-sm text-amber-300 mb-2 font-medium">💡 Follow-up suggestion:</p>
                <button
                  onClick={useFollowUp}
                  className="text-left w-full p-2 bg-amber-500/20 hover:bg-amber-500/30 rounded border border-amber-500/30 transition-colors"
                >
                  <p className="text-sm text-amber-100">{followUp}</p>
                </button>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="bg-slate-800/50 backdrop-blur-xl border-t border-slate-700/50 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask a legal question... (e.g., 'What is Section 498A IPC?')"
                className="flex-1 bg-slate-700/50 border border-slate-600 rounded-xl px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-transparent transition-all"
                rows={2}
                disabled={isLoading}
              />
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim() || isLoading}
                className="px-6 bg-gradient-to-r from-amber-500 to-orange-500 hover:from-amber-600 hover:to-orange-600 disabled:from-slate-600 disabled:to-slate-600 disabled:cursor-not-allowed rounded-xl flex items-center justify-center transition-all shadow-lg hover:shadow-xl"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
            <p className="text-xs text-slate-500 mt-2 text-center">
              Powered by Adaptive RAG with Multi-domain Routing • Press Enter to send
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

// Simple Markdown Component
const ReactMarkdown = ({ content }) => {
  const formatText = (text) => {
    // Bold
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Italic
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    // Code
    text = text.replace(/`(.+?)`/g, '<code class="bg-slate-900/50 px-1.5 py-0.5 rounded text-amber-300">$1</code>');
    // Line breaks
    text = text.replace(/\n/g, '<br />');
    return text;
  };

  const parseMarkdown = () => {
    const lines = content.split('\n');
    const elements = [];
    let currentList = [];
    let listType = null;

    lines.forEach((line, idx) => {
      // Headers
      if (line.startsWith('### ')) {
        if (currentList.length > 0) {
          elements.push(
            <ul key={`list-${idx}`} className="list-disc list-inside space-y-1 ml-4">
              {currentList.map((item, i) => (
                <li key={i} dangerouslySetInnerHTML={{ __html: formatText(item) }} />
              ))}
            </ul>
          );
          currentList = [];
        }
        elements.push(
          <h3 key={idx} className="text-lg font-bold text-amber-400 mt-4 mb-2">
            {line.replace('### ', '')}
          </h3>
        );
      } else if (line.startsWith('## ')) {
        elements.push(
          <h2 key={idx} className="text-xl font-bold text-amber-400 mt-4 mb-2">
            {line.replace('## ', '')}
          </h2>
        );
      } else if (line.startsWith('# ')) {
        elements.push(
          <h1 key={idx} className="text-2xl font-bold text-amber-400 mt-4 mb-2">
            {line.replace('# ', '')}
          </h1>
        );
      } else if (line.match(/^[\-\*]\s/)) {
        // Bullet list
        currentList.push(line.replace(/^[\-\*]\s/, ''));
        listType = 'ul';
      } else if (line.match(/^\d+\.\s/)) {
        // Numbered list
        currentList.push(line.replace(/^\d+\.\s/, ''));
        listType = 'ol';
      } else if (line.trim()) {
        if (currentList.length > 0) {
          const ListTag = listType === 'ol' ? 'ol' : 'ul';
          elements.push(
            <ListTag key={`list-${idx}`} className={`${listType === 'ol' ? 'list-decimal' : 'list-disc'} list-inside space-y-1 ml-4`}>
              {currentList.map((item, i) => (
                <li key={i} dangerouslySetInnerHTML={{ __html: formatText(item) }} />
              ))}
            </ListTag>
          );
          currentList = [];
          listType = null;
        }
        elements.push(
          <p key={idx} dangerouslySetInnerHTML={{ __html: formatText(line) }} className="mb-2" />
        );
      }
    });

    if (currentList.length > 0) {
      const ListTag = listType === 'ol' ? 'ol' : 'ul';
      elements.push(
        <ListTag key="list-final" className={`${listType === 'ol' ? 'list-decimal' : 'list-disc'} list-inside space-y-1 ml-4`}>
          {currentList.map((item, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: formatText(item) }} />
          ))}
        </ListTag>
      );
    }

    return elements;
  };

  return <div className="markdown-content">{parseMarkdown()}</div>;
};

export default LegalAssistChat;