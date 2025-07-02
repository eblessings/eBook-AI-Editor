import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Sparkles, Check, X, Loader2, Sun, Moon, Copy, FileText,
  Bold, Italic, Underline, Link, AlignLeft, AlignCenter, 
  AlignRight, List, ListOrdered, IndentDecrease, IndentIncrease,
  Palette, MoveVertical, Upload, Download, Settings, BarChart3,
  BookOpen, Zap, Brain, Eye, Target, TrendingUp, Save,
  RefreshCw, ChevronDown, Play, Pause, Users, Globe,
  Filter, Search, Layout, Type, Image, Server, Cpu,
  Cloud, Key, Database, Monitor, AlertCircle, CheckCircle,
  ExternalLink, Sliders, Wifi, WifiOff
} from 'lucide-react';

// Translation system (simplified)
const translations = {
  "en-US": {
    "appTitle": "eBook Editor Pro",
    "yourText": "Your Text",
    "sample": "Sample",
    "copy": "Copy",
    "fontFamily": "Font Family",
    "fontSize": "Font Size",
    "bold": "Bold",
    "italic": "Italic",
    "underline": "Underline",
    "textColor": "Text Color",
    "addLink": "Add Link",
    "alignLeft": "Align Left",
    "alignCenter": "Align Center",
    "alignRight": "Align Right",
    "lineSpacing": "Line Spacing",
    "bulletList": "Bullet List",
    "numberedList": "Numbered List",
    "analyzeText": "Analyze Text",
    "analyzing": "Analyzing...",
    "suggestions": "Suggestions",
    "all": "All",
    "grammar": "Grammar",
    "spelling": "Spelling",
    "punctuation": "Punctuation",
    "style": "Style",
    "clarity": "Clarity",
    "accept": "Accept",
    "reject": "Reject",
    "cancel": "Cancel"
  }
};

const t = (key) => translations["en-US"][key] || key;

const EBookEditor = () => {
  // Core state
  const [text, setText] = useState('');
  const [htmlContent, setHtmlContent] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState('');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [activeCategory, setActiveCategory] = useState('all');
  const [currentView, setCurrentView] = useState('editor');
  const [showTooltip, setShowTooltip] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  
  // Editor state
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [showLinkDialog, setShowLinkDialog] = useState(false);
  const [linkUrl, setLinkUrl] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  
  // Enhanced AI Configuration state
  const [aiConfig, setAiConfig] = useState({
    type: 'local', // 'local', 'external', 'claude', 'mistral', 'openai'
    model: 'microsoft/DialoGPT-medium',
    endpoint: '',
    apiKey: '',
    isConnected: false,
    status: 'disconnected'
  });
  
  const [showAiConfig, setShowAiConfig] = useState(false);
  const [availableModels, setAvailableModels] = useState({
    local: ['microsoft/DialoGPT-small', 'microsoft/DialoGPT-medium', 'microsoft/DialoGPT-large'],
    external: [],
    claude: ['claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229'],
    mistral: ['mistral-tiny', 'mistral-small', 'mistral-medium', 'mistral-large'],
    openai: ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo-preview']
  });
  
  // Project state
  const [projectMetadata, setProjectMetadata] = useState({
    title: 'Untitled Book',
    author: 'Unknown Author',
    description: '',
    genre: '',
    language: 'en'
  });
  
  // Analytics state
  const [analytics, setAnalytics] = useState({
    wordCount: 0,
    characterCount: 0,
    readabilityScore: 75,
    sessionTime: 0
  });
  
  // Settings state
  const [settings, setSettings] = useState({
    realTimeAnalysis: true,
    autoSave: true,
    aiType: 'local',
    dailyGoal: 1000
  });

  const editorRef = useRef(null);

  // Configuration
  const categories = [
    { id: 'all', label: t('all'), color: 'bg-purple-500', icon: Target },
    { id: 'grammar', label: t('grammar'), color: 'bg-blue-500', icon: Check },
    { id: 'spelling', label: t('spelling'), color: 'bg-red-500', icon: X },
    { id: 'style', label: t('style'), color: 'bg-green-500', icon: Sparkles },
    { id: 'clarity', label: t('clarity'), color: 'bg-indigo-500', icon: Eye }
  ];

  const exportFormats = [
    { id: 'epub', label: 'EPUB', description: 'Standard eBook format', icon: BookOpen },
    { id: 'pdf', label: 'PDF', description: 'Portable Document Format', icon: FileText },
    { id: 'docx', label: 'DOCX', description: 'Microsoft Word Document', icon: FileText },
    { id: 'html', label: 'HTML', description: 'Web Page Format', icon: Globe },
    { id: 'txt', label: 'TXT', description: 'Plain Text Format', icon: Type }
  ];

  const aiProviders = [
    { 
      id: 'local', 
      name: 'Local Models', 
      icon: Cpu, 
      description: 'Run AI models locally on your machine',
      endpoint: '',
      requiresKey: false
    },
    { 
      id: 'claude', 
      name: 'Anthropic Claude', 
      icon: Brain, 
      description: 'Advanced AI with excellent reasoning',
      endpoint: 'https://api.anthropic.com/v1',
      requiresKey: true
    },
    { 
      id: 'mistral', 
      name: 'Mistral AI', 
      icon: Server, 
      description: 'Fast and efficient European AI',
      endpoint: 'https://api.mistral.ai/v1',
      requiresKey: true
    },
    { 
      id: 'openai', 
      name: 'OpenAI GPT', 
      icon: Sparkles, 
      description: 'Powerful language models from OpenAI',
      endpoint: 'https://api.openai.com/v1',
      requiresKey: true
    },
    { 
      id: 'external', 
      name: 'Custom Endpoint', 
      icon: Cloud, 
      description: 'Connect to your own AI server',
      endpoint: 'http://localhost:1234/v1',
      requiresKey: false
    }
  ];

  const sampleTexts = [
    `Chapter 1: The Digital Revolution

The world has changed more in the past two decades than in the previous two centuries combined. We stand at the precipice of a new era, where artificial intelligence and human creativity converge to create unprecedented possibilities.

In the not-so-distant past, the idea of machines that could think, learn, and create was relegated to the realm of science fiction. Today, these technologies are not just real—they're reshaping every aspect of our lives.

This book explores the fascinating intersection of human ingenuity and artificial intelligence, examining how we can harness these powerful tools while preserving what makes us uniquely human.`,

    `The Art of Mindful Writing

Writing is more than just putting words on a page. It's an act of discovery, a conversation with your deepest thoughts, and a bridge between your inner world and the reality you wish to create.

In our fast-paced digital age, the art of mindful writing has become more important than ever. When we slow down, breathe deeply, and connect with our authentic voice, we unlock a wellspring of creativity.

Whether you're crafting a novel, writing business communications, or simply journaling your thoughts, mindful writing can transform not just your words, but your entire relationship with communication and self-expression.`
  ];

  // Update content function
  const updateContent = useCallback(() => {
    if (!editorRef.current) return;
    
    const htmlContent = editorRef.current.innerHTML;
    setHtmlContent(htmlContent);
    
    // Extract plain text
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;
    const plainText = tempDiv.textContent || tempDiv.innerText || '';
    setText(plainText);
    
    // Update analytics
    const words = plainText.trim().split(/\s+/).filter(word => word.length > 0);
    setAnalytics(prev => ({
      ...prev,
      wordCount: words.length,
      characterCount: plainText.length
    }));
  }, []);

  // Format text function
  const formatText = useCallback((command, value = null) => {
    if (!editorRef.current) return;
    
    editorRef.current.focus();
    document.execCommand(command, false, value);
    updateContent();
  }, [updateContent]);

  // Load sample text
  const loadSampleText = useCallback(() => {
    if (!editorRef.current) return;
    
    const randomSample = sampleTexts[Math.floor(Math.random() * sampleTexts.length)];
    const formattedContent = randomSample
      .split('\n\n')
      .map(paragraph => `<div style="margin-bottom: 1em;">${paragraph.replace(/\n/g, '<br>')}</div>`)
      .join('');
    
    editorRef.current.innerHTML = formattedContent;
    updateContent();
  }, [updateContent]);

  // Copy text function
  const copyText = useCallback(() => {
    if (!text) return;
    
    navigator.clipboard.writeText(text).then(() => {
      setError('Text copied to clipboard!');
      setTimeout(() => setError(''), 2000);
    });
  }, [text]);

  // Enhanced AI configuration function
  const configureAI = useCallback(async (config) => {
    setAiConfig(prev => ({ ...prev, status: 'connecting' }));
    
    try {
      const response = await fetch('/api/configure-ai', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ai_type: config.type === 'local' ? 'local' : 'external',
          model_name: config.model,
          api_endpoint: config.endpoint,
          api_key: config.apiKey
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to configure AI');
      }

      const result = await response.json();
      
      setAiConfig(prev => ({
        ...prev,
        isConnected: result.is_ready,
        status: result.is_ready ? 'connected' : 'error'
      }));

      setError(result.is_ready ? 'AI configured successfully!' : 'AI configuration failed');
      setTimeout(() => setError(''), 3000);

    } catch (error) {
      console.error('AI configuration error:', error);
      setAiConfig(prev => ({ ...prev, status: 'error', isConnected: false }));
      setError('Failed to configure AI: ' + error.message);
      setTimeout(() => setError(''), 3000);
    }
  }, []);

  // Test AI connection
  const testAIConnection = useCallback(async () => {
    if (!aiConfig.type) return;
    
    setAiConfig(prev => ({ ...prev, status: 'testing' }));
    
    try {
      const response = await fetch('/api/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [{ role: 'user', content: 'Hello, please respond with just "AI is working"' }],
          max_tokens: 10
        }),
      });

      if (response.ok) {
        setAiConfig(prev => ({ ...prev, status: 'connected', isConnected: true }));
        setError('AI connection successful!');
      } else {
        setAiConfig(prev => ({ ...prev, status: 'error', isConnected: false }));
        setError('AI connection failed');
      }
    } catch (error) {
      setAiConfig(prev => ({ ...prev, status: 'error', isConnected: false }));
      setError('AI connection error: ' + error.message);
    }
    
    setTimeout(() => setError(''), 3000);
  }, [aiConfig.type]);

  // Enhanced analyze text function
  const analyzeText = useCallback(async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError('');
    setSuggestions([]);

    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          include_ai_suggestions: true,
          suggestion_categories: ['grammar', 'spelling', 'style', 'clarity'],
          language: 'en-US',
          readability_metrics: true,
          advanced_analysis: true
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      // Transform the response to match our expected format
      const transformedSuggestions = [];
      
      // Add grammar issues
      if (result.grammar_issues) {
        result.grammar_issues.forEach(issue => {
          transformedSuggestions.push({
            category: 'grammar',
            issue: issue.message,
            suggestion: issue.suggestions[0] || 'Fix grammar issue',
            explanation: `Grammar issue: ${issue.category}`,
            confidence: 0.8,
            position: issue.offset || 0,
            length: issue.length || 10
          });
        });
      }
      
      // Add spelling errors
      if (result.spelling_errors) {
        result.spelling_errors.forEach(error => {
          transformedSuggestions.push({
            category: 'spelling',
            issue: error.word,
            suggestion: error.suggestions[0] || 'Correct spelling',
            explanation: 'Possible spelling error',
            confidence: error.confidence || 0.7,
            position: error.offset || 0,
            length: error.word?.length || 5
          });
        });
      }
      
      // Add style suggestions
      if (result.style_suggestions) {
        result.style_suggestions.forEach(suggestion => {
          transformedSuggestions.push({
            category: 'style',
            issue: suggestion.original_text || 'Style improvement',
            suggestion: suggestion.suggested_text || suggestion.message,
            explanation: suggestion.explanation,
            confidence: suggestion.confidence || 0.6,
            position: 0,
            length: 10
          });
        });
      }
      
      // Add AI suggestions if available
      if (result.ai_suggestions) {
        result.ai_suggestions.forEach(suggestion => {
          transformedSuggestions.push(suggestion);
        });
      }

      setSuggestions(transformedSuggestions);
      
      // Update analytics with readability scores
      if (result.readability_scores) {
        setAnalytics(prev => ({
          ...prev,
          readabilityScore: Math.round(result.readability_scores.flesch_reading_ease || prev.readabilityScore)
        }));
      }

    } catch (error) {
      console.error('Analysis error:', error);
      setError('Analysis failed: ' + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  }, [text]);

  // Apply suggestion
  const applySuggestion = useCallback((suggestion) => {
    if (!editorRef.current) return;
    
    const content = editorRef.current.innerHTML;
    const updatedContent = content.replace(suggestion.issue, suggestion.suggestion);
    editorRef.current.innerHTML = updatedContent;
    updateContent();
    
    setSuggestions(prev => prev.filter(s => s !== suggestion));
    
    setError('Suggestion applied!');
    setTimeout(() => setError(''), 2000);
  }, [updateContent]);

  // Dismiss suggestion
  const dismissSuggestion = useCallback((suggestion) => {
    setSuggestions(prev => prev.filter(s => s !== suggestion));
  }, []);

  // Enhanced file upload
  const handleFileUpload = useCallback(async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadProgress(10);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('enhance_with_ai', 'false');
    formData.append('target_format', 'epub');
    
    try {
      setUploadProgress(30);
      
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      
      setUploadProgress(70);
      
      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }
      
      const result = await response.json();
      
      setUploadProgress(90);
      
      if (result.text_content && editorRef.current) {
        const formattedContent = result.text_content
          .split('\n\n')
          .map(paragraph => `<div style="margin-bottom: 1em;">${paragraph.replace(/\n/g, '<br>')}</div>`)
          .join('');
        
        editorRef.current.innerHTML = formattedContent;
        updateContent();
        
        // Update project metadata if available
        if (result.analysis && result.analysis.metadata) {
          setProjectMetadata(prev => ({
            ...prev,
            title: result.analysis.metadata.title || prev.title,
            author: result.analysis.metadata.author || prev.author
          }));
        }
      }
      
      setUploadProgress(100);
      setError(`File uploaded successfully! ${result.word_count} words extracted.`);
      setTimeout(() => {
        setUploadProgress(0);
        setError('');
      }, 2000);
      
    } catch (error) {
      console.error('Upload error:', error);
      setError('Upload failed: ' + error.message);
      setUploadProgress(0);
      setTimeout(() => setError(''), 3000);
    }
  }, [updateContent]);

  // Enhanced eBook generation
  const generateEBook = useCallback(async (format) => {
    if (!text.trim()) {
      setError('Please enter some content to generate eBook');
      return;
    }

    setIsGenerating(true);
    setError('');

    try {
      const response = await fetch('/api/generate-ebook', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: text,
          format: format,
          metadata: {
            title: projectMetadata.title,
            author: projectMetadata.author,
            description: projectMetadata.description,
            publisher: 'eBook Editor Pro',
            language: projectMetadata.language,
            genre: projectMetadata.genre
          },
          format_options: {
            font_family: 'Georgia',
            font_size: 12,
            line_height: 1.5,
            margin_top: 20,
            margin_bottom: 20,
            margin_left: 20,
            margin_right: 20,
            page_break_before_chapter: true,
            include_toc: true,
            include_cover: true,
            justify_text: true,
            format: format
          },
          ai_enhancement_options: {
            enhance_before_generation: false,
            improve_grammar: true,
            enhance_style: false,
            auto_correct_spelling: true,
            improvement_readability: false,
            enhancement_strength: 'moderate'
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Generation failed: ${response.status}`);
      }

      // Handle file download
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `${projectMetadata.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      setError(`${format.toUpperCase()} eBook generated successfully!`);
      setTimeout(() => setError(''), 3000);

    } catch (error) {
      console.error('Generation error:', error);
      setError('eBook generation failed: ' + error.message);
      setTimeout(() => setError(''), 3000);
    } finally {
      setIsGenerating(false);
    }
  }, [text, projectMetadata]);

  // Initialize editor
  useEffect(() => {
    if (editorRef.current && !editorRef.current.innerHTML) {
      editorRef.current.innerHTML = '<div style="margin-bottom: 1em;">Start writing your book here...</div>';
    }
  }, []);

  // Auto-save functionality
  useEffect(() => {
    if (!settings.autoSave || !text.trim()) return;

    const timeoutId = setTimeout(() => {
      localStorage.setItem('ebook-editor-content', text);
      localStorage.setItem('ebook-editor-metadata', JSON.stringify(projectMetadata));
      localStorage.setItem('ebook-editor-ai-config', JSON.stringify(aiConfig));
    }, 3000);

    return () => clearTimeout(timeoutId);
  }, [text, projectMetadata, settings.autoSave, aiConfig]);

  // Load saved content
  useEffect(() => {
    const savedContent = localStorage.getItem('ebook-editor-content');
    const savedMetadata = localStorage.getItem('ebook-editor-metadata');
    const savedAiConfig = localStorage.getItem('ebook-editor-ai-config');
    
    if (savedContent && editorRef.current) {
      const formattedContent = savedContent
        .split('\n\n')
        .map(paragraph => `<div style="margin-bottom: 1em;">${paragraph.replace(/\n/g, '<br>')}</div>`)
        .join('');
      editorRef.current.innerHTML = formattedContent;
      updateContent();
    }
    
    if (savedMetadata) {
      try {
        setProjectMetadata(JSON.parse(savedMetadata));
      } catch (e) {
        console.error('Failed to parse saved metadata');
      }
    }
    
    if (savedAiConfig) {
      try {
        setAiConfig(JSON.parse(savedAiConfig));
      } catch (e) {
        console.error('Failed to parse saved AI config');
      }
    }
  }, [updateContent]);

  // Filter suggestions
  const filteredSuggestions = activeCategory === 'all' 
    ? suggestions 
    : suggestions.filter(s => s.category === activeCategory);

  // Component renders
  const renderToolbar = () => (
    <div className={`flex flex-wrap items-center gap-2 p-3 mb-4 rounded-lg border ${
      isDarkMode ? 'bg-gray-900 border-gray-700' : 'bg-gray-50 border-gray-300'
    }`}>
      <button
        onClick={() => formatText('bold')}
        className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-700'
        }`}
        title={t('bold')}
      >
        <Bold className="w-4 h-4" />
      </button>
      
      <button
        onClick={() => formatText('italic')}
        className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-700'
        }`}
        title={t('italic')}
      >
        <Italic className="w-4 h-4" />
      </button>
      
      <button
        onClick={() => formatText('underline')}
        className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-700'
        }`}
        title={t('underline')}
      >
        <Underline className="w-4 h-4" />
      </button>
      
      <div className={`w-px h-6 ${isDarkMode ? 'bg-gray-700' : 'bg-gray-300'}`} />
      
      <button
        onClick={() => formatText('justifyLeft')}
        className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-700'
        }`}
        title={t('alignLeft')}
      >
        <AlignLeft className="w-4 h-4" />
      </button>
      
      <button
        onClick={() => formatText('justifyCenter')}
        className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-700'
        }`}
        title={t('alignCenter')}
      >
        <AlignCenter className="w-4 h-4" />
      </button>
      
      <button
        onClick={() => formatText('insertUnorderedList')}
        className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 ${
          isDarkMode ? 'text-gray-300' : 'text-gray-700'
        }`}
        title={t('bulletList')}
      >
        <List className="w-4 h-4" />
      </button>
    </div>
  );

  const renderAIConfiguration = () => (
    <div className={`rounded-xl shadow-lg p-6 mb-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <div className="flex justify-between items-center mb-4">
        <h3 className={`text-xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          AI Configuration
        </h3>
        <div className="flex items-center gap-2">
          <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
            aiConfig.status === 'connected' ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300' :
            aiConfig.status === 'connecting' || aiConfig.status === 'testing' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' :
            'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
          }`}>
            {aiConfig.status === 'connected' ? <CheckCircle className="w-4 h-4" /> :
             aiConfig.status === 'connecting' || aiConfig.status === 'testing' ? <Loader2 className="w-4 h-4 animate-spin" /> :
             <AlertCircle className="w-4 h-4" />}
            {aiConfig.status === 'connected' ? 'Connected' :
             aiConfig.status === 'connecting' ? 'Connecting...' :
             aiConfig.status === 'testing' ? 'Testing...' :
             'Disconnected'}
          </div>
          <button
            onClick={() => setShowAiConfig(!showAiConfig)}
            className={`p-2 rounded-lg ${isDarkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-200'}`}
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>

      {showAiConfig && (
        <div className="space-y-4">
          {/* AI Provider Selection */}
          <div>
            <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
              AI Provider
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {aiProviders.map((provider) => (
                <div
                  key={provider.id}
                  onClick={() => setAiConfig(prev => ({ 
                    ...prev, 
                    type: provider.id,
                    endpoint: provider.endpoint,
                    model: availableModels[provider.id]?.[0] || ''
                  }))}
                  className={`p-3 rounded-lg border-2 cursor-pointer transition-all ${
                    aiConfig.type === provider.id
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                      : isDarkMode 
                        ? 'border-gray-700 hover:border-gray-600 bg-gray-900'
                        : 'border-gray-300 hover:border-gray-400 bg-gray-50'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    <provider.icon className="w-5 h-5" />
                    <span className="font-medium">{provider.name}</span>
                    {provider.requiresKey && <Key className="w-4 h-4 text-gray-500" />}
                  </div>
                  <p className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {provider.description}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Model Selection */}
          {aiConfig.type && availableModels[aiConfig.type] && (
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                Model
              </label>
              <select
                value={aiConfig.model}
                onChange={(e) => setAiConfig(prev => ({ ...prev, model: e.target.value }))}
                className={`w-full px-3 py-2 rounded-lg border ${
                  isDarkMode 
                    ? 'bg-gray-900 border-gray-700 text-white' 
                    : 'bg-gray-50 border-gray-300 text-gray-900'
                }`}
              >
                {availableModels[aiConfig.type].map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>
          )}

          {/* API Endpoint */}
          {aiConfig.type !== 'local' && (
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                API Endpoint
              </label>
              <input
                type="text"
                value={aiConfig.endpoint}
                onChange={(e) => setAiConfig(prev => ({ ...prev, endpoint: e.target.value }))}
                placeholder="https://api.example.com/v1"
                className={`w-full px-3 py-2 rounded-lg border ${
                  isDarkMode 
                    ? 'bg-gray-900 border-gray-700 text-white' 
                    : 'bg-gray-50 border-gray-300 text-gray-900'
                }`}
              />
            </div>
          )}

          {/* API Key */}
          {aiProviders.find(p => p.id === aiConfig.type)?.requiresKey && (
            <div>
              <label className={`block text-sm font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                API Key
              </label>
              <input
                type="password"
                value={aiConfig.apiKey}
                onChange={(e) => setAiConfig(prev => ({ ...prev, apiKey: e.target.value }))}
                placeholder="Enter your API key"
                className={`w-full px-3 py-2 rounded-lg border ${
                  isDarkMode 
                    ? 'bg-gray-900 border-gray-700 text-white' 
                    : 'bg-gray-50 border-gray-300 text-gray-900'
                }`}
              />
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-3">
            <button
              onClick={() => configureAI(aiConfig)}
              disabled={aiConfig.status === 'connecting'}
              className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                aiConfig.status === 'connecting'
                  ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              {aiConfig.status === 'connecting' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              {aiConfig.status === 'connecting' ? 'Connecting...' : 'Save Configuration'}
            </button>
            
            <button
              onClick={testAIConnection}
              disabled={aiConfig.status === 'testing' || !aiConfig.type}
              className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                aiConfig.status === 'testing' || !aiConfig.type
                  ? 'bg-gray-400 text-gray-200 cursor-not-allowed'
                  : 'bg-green-500 hover:bg-green-600 text-white'
              }`}
            >
              {aiConfig.status === 'testing' ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Wifi className="w-4 h-4" />
              )}
              {aiConfig.status === 'testing' ? 'Testing...' : 'Test Connection'}
            </button>
          </div>
        </div>
      )}
    </div>
  );

  const renderSuggestions = () => (
    <div className="space-y-3 max-h-96 overflow-y-auto">
      {filteredSuggestions.length === 0 ? (
        <div className={`text-center py-12 ${isDarkMode ? 'text-gray-500' : 'text-gray-400'}`}>
          {suggestions.length === 0 
            ? 'Click "Analyze Text" to get AI suggestions'
            : 'No suggestions in this category'}
        </div>
      ) : (
        filteredSuggestions.map((suggestion, index) => (
          <div
            key={index}
            className={`p-4 rounded-lg border transition-all hover:shadow-md ${
              isDarkMode 
                ? 'bg-gray-900 border-gray-700 hover:border-gray-600' 
                : 'bg-gray-50 border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="flex justify-between items-start mb-2">
              <span className={`inline-block px-2 py-1 rounded-full text-xs font-medium text-white ${
                categories.find(c => c.id === suggestion.category)?.color || 'bg-gray-500'
              }`}>
                {suggestion.category}
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => applySuggestion(suggestion)}
                  className="p-1 rounded hover:bg-green-500/20 text-green-500 transition-colors"
                  title="Apply suggestion"
                >
                  <Check className="w-4 h-4" />
                </button>
                <button
                  onClick={() => dismissSuggestion(suggestion)}
                  className="p-1 rounded hover:bg-red-500/20 text-red-500 transition-colors"
                  title="Dismiss"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <span className={`line-through ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                  {suggestion.issue}
                </span>
                <span className={isDarkMode ? 'text-gray-500' : 'text-gray-400'}>→</span>
                <span className={`font-medium ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                  {suggestion.suggestion}
                </span>
              </div>
              <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                {suggestion.explanation}
              </p>
              {suggestion.confidence && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">Confidence:</span>
                  <div className="flex-1 bg-gray-200 rounded-full h-1">
                    <div 
                      className="bg-blue-500 h-1 rounded-full" 
                      style={{ width: `${suggestion.confidence * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500">{Math.round(suggestion.confidence * 100)}%</span>
                </div>
              )}
            </div>
          </div>
        ))
      )}
    </div>
  );

  // Main render
  return (
    <div className={`min-h-screen transition-colors duration-300 ${
      isDarkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      {/* Header */}
      <div className={`sticky top-0 z-50 border-b ${
        isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'
      }`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center gap-3">
              <BookOpen className={`w-8 h-8 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
              <div>
                <h1 className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                  eBook Editor Pro
                </h1>
                <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                  AI-Powered Professional eBook Creation
                </p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
                aiConfig.isConnected
                  ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                  : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
              }`}>
                {aiConfig.isConnected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
                AI: {aiConfig.isConnected ? 'Ready' : 'Disconnected'}
              </div>

              <button
                onClick={() => setIsDarkMode(!isDarkMode)}
                className={`p-2 rounded-lg transition-colors ${
                  isDarkMode ? 'bg-gray-700 hover:bg-gray-600 text-yellow-400' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                }`}
              >
                {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
              </button>
            </div>
          </div>

          {/* Navigation */}
          <div className="flex gap-4 pb-4">
            {[
              { id: 'editor', label: 'Editor', icon: Type },
              { id: 'analytics', label: 'Analytics', icon: BarChart3 },
              { id: 'settings', label: 'Settings', icon: Settings },
              { id: 'export', label: 'Export', icon: Download }
            ].map(item => (
              <button
                key={item.id}
                onClick={() => setCurrentView(item.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                  currentView === item.id
                    ? isDarkMode ? 'bg-purple-600 text-white' : 'bg-purple-500 text-white'
                    : isDarkMode 
                      ? 'hover:bg-gray-700 text-gray-400 hover:text-white' 
                      : 'hover:bg-gray-200 text-gray-600 hover:text-gray-900'
                }`}
              >
                <item.icon className="w-4 h-4" />
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto p-4 md:p-8">
        {currentView === 'editor' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Editor Panel */}
            <div className={`lg:col-span-2 space-y-6`}>
              {/* AI Configuration */}
              {renderAIConfiguration()}
              
              {/* Editor */}
              <div className={`rounded-xl shadow-lg p-6 ${
                isDarkMode ? 'bg-gray-800' : 'bg-white'
              }`}>
                <div className="flex justify-between items-center mb-4">
                  <h2 className={`text-xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {projectMetadata.title}
                  </h2>
                  <div className="flex gap-2">
                    <input
                      type="file"
                      id="file-upload"
                      className="hidden"
                      accept=".txt,.docx,.pdf,.epub,.md,.html"
                      onChange={handleFileUpload}
                    />
                    <button
                      onClick={() => document.getElementById('file-upload')?.click()}
                      className={`px-3 py-2 text-sm rounded-lg transition-colors flex items-center gap-1 ${
                        isDarkMode 
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                      }`}
                    >
                      <Upload className="w-4 h-4" />
                      Upload
                    </button>
                    
                    <button
                      onClick={loadSampleText}
                      className={`px-3 py-2 text-sm rounded-lg transition-colors flex items-center gap-1 ${
                        isDarkMode 
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                      }`}
                    >
                      <FileText className="w-4 h-4" />
                      Sample
                    </button>
                    
                    <button
                      onClick={copyText}
                      className={`px-3 py-2 text-sm rounded-lg transition-colors flex items-center gap-1 ${
                        isDarkMode 
                          ? 'bg-gray-700 hover:bg-gray-600 text-gray-300' 
                          : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                      }`}
                    >
                      <Copy className="w-4 h-4" />
                      Copy
                    </button>
                  </div>
                </div>

                {/* Upload Progress */}
                {uploadProgress > 0 && uploadProgress < 100 && (
                  <div className="mb-4">
                    <div className="flex justify-between text-sm mb-2">
                      <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>Uploading...</span>
                      <span className={isDarkMode ? 'text-gray-400' : 'text-gray-500'}>{uploadProgress}%</span>
                    </div>
                    <div className={`w-full h-2 rounded-full ${isDarkMode ? 'bg-gray-700' : 'bg-gray-200'}`}>
                      <div 
                        className="h-2 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-300"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Toolbar */}
                {renderToolbar()}
                
                {/* Editor */}
                <div
                  ref={editorRef}
                  contentEditable={true}
                  suppressContentEditableWarning={true}
                  onInput={updateContent}
                  className={`w-full h-96 p-4 rounded-lg border transition-colors overflow-y-auto focus:outline-none focus:ring-2 ${
                    isDarkMode 
                      ? 'bg-gray-900 border-gray-700 text-white focus:ring-purple-500' 
                      : 'bg-gray-50 border-gray-300 text-gray-900 focus:ring-purple-400'
                  }`}
                  style={{ 
                    minHeight: '24rem',
                    fontFamily: 'Georgia, serif',
                    fontSize: '16px',
                    lineHeight: '1.6'
                  }}
                />
                
                {/* Status Bar */}
                <div className="mt-4 flex justify-between items-center">
                  <div className="flex gap-4 text-sm">
                    <span className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {analytics.wordCount} words
                    </span>
                    <span className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {analytics.characterCount} characters
                    </span>
                    <span className={`${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Readability: {analytics.readabilityScore}/100
                    </span>
                  </div>
                  
                  <div className="flex gap-2">
                    <button
                      onClick={analyzeText}
                      disabled={isAnalyzing || !text.trim() || !aiConfig.isConnected}
                      className={`px-6 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                        isAnalyzing || !text.trim() || !aiConfig.isConnected
                          ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                          : 'bg-gradient-to-r from-purple-500 to-indigo-600 text-white hover:from-purple-600 hover:to-indigo-700 shadow-lg'
                      }`}
                    >
                      {isAnalyzing ? (
                        <>
                          <Loader2 className="w-4 h-4 animate-spin" />
                          {t('analyzing')}
                        </>
                      ) : (
                        <>
                          <Brain className="w-4 h-4" />
                          Analyze with AI
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {error && (
                  <div className={`mt-4 p-3 rounded-lg ${
                    error.includes('successfully') || error.includes('copied') 
                      ? 'bg-green-500/10 border border-green-500/20 text-green-500'
                      : 'bg-red-500/10 border border-red-500/20 text-red-500'
                  }`}>
                    <p className="text-sm">{error}</p>
                  </div>
                )}
              </div>
            </div>

            {/* Suggestions Panel */}
            <div className={`lg:col-span-1 rounded-xl shadow-lg p-6 ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <div className="flex justify-between items-center mb-4">
                <h2 className={`text-xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                  AI Suggestions
                </h2>
                {isAnalyzing && (
                  <Loader2 className="w-5 h-5 animate-spin text-purple-500" />
                )}
              </div>

              {/* Category Filter */}
              <div className="flex flex-wrap gap-2 mb-4">
                {categories.map(category => (
                  <button
                    key={category.id}
                    onClick={() => setActiveCategory(category.id)}
                    className={`flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium transition-all ${
                      activeCategory === category.id
                        ? `${category.color} text-white`
                        : isDarkMode 
                          ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' 
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    <category.icon className="w-3 h-3" />
                    {category.label}
                    {suggestions.filter(s => category.id === 'all' || s.category === category.id).length > 0 && (
                      <span className="ml-1">
                        ({suggestions.filter(s => category.id === 'all' || s.category === category.id).length})
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {renderSuggestions()}
            </div>
          </div>
        )}

        {/* Export View */}
        {currentView === 'export' && (
          <div className="space-y-6">
            <div className={`p-6 rounded-xl shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h3 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                Export eBook
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {exportFormats.map((format) => (
                  <div
                    key={format.id}
                    onClick={() => generateEBook(format.id)}
                    className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                      isGenerating
                        ? 'opacity-50 cursor-not-allowed'
                        : isDarkMode
                          ? 'border-gray-700 hover:border-gray-600 bg-gray-900 hover:bg-gray-800'
                          : 'border-gray-300 hover:border-gray-400 bg-gray-50 hover:bg-gray-100'
                    }`}
                  >
                    <format.icon className={`w-8 h-8 mx-auto mb-2 ${
                      isDarkMode ? 'text-gray-400' : 'text-gray-600'
                    }`} />
                    <h4 className={`font-medium text-center ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                      {format.label}
                    </h4>
                    <p className={`text-sm text-center mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      {format.description}
                    </p>
                  </div>
                ))}
              </div>
              
              {isGenerating && (
                <div className="mt-6 flex items-center justify-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin text-purple-500" />
                  <span className={isDarkMode ? 'text-gray-300' : 'text-gray-700'}>
                    Generating eBook...
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Settings View */}
        {currentView === 'settings' && (
          <div className="space-y-6">
            <div className={`p-6 rounded-xl shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h3 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                Book Information
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className={`block text-sm font-medium mb-1 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    Title
                  </label>
                  <input
                    type="text"
                    value={projectMetadata.title}
                    onChange={(e) => setProjectMetadata(prev => ({ ...prev, title: e.target.value }))}
                    className={`w-full px-3 py-2 rounded-lg border ${
                      isDarkMode 
                        ? 'bg-gray-900 border-gray-700 text-white' 
                        : 'bg-gray-50 border-gray-300 text-gray-900'
                    }`}
                    placeholder="Enter book title"
                  />
                </div>
                <div>
                  <label className={`block text-sm font-medium mb-1 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    Author
                  </label>
                  <input
                    type="text"
                    value={projectMetadata.author}
                    onChange={(e) => setProjectMetadata(prev => ({ ...prev, author: e.target.value }))}
                    className={`w-full px-3 py-2 rounded-lg border ${
                      isDarkMode 
                        ? 'bg-gray-900 border-gray-700 text-white' 
                        : 'bg-gray-50 border-gray-300 text-gray-900'
                    }`}
                    placeholder="Enter author name"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className={`block text-sm font-medium mb-1 ${
                    isDarkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    Description
                  </label>
                  <textarea
                    value={projectMetadata.description}
                    onChange={(e) => setProjectMetadata(prev => ({ ...prev, description: e.target.value }))}
                    className={`w-full px-3 py-2 rounded-lg border ${
                      isDarkMode 
                        ? 'bg-gray-900 border-gray-700 text-white' 
                        : 'bg-gray-50 border-gray-300 text-gray-900'
                    }`}
                    rows="3"
                    placeholder="Enter book description"
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Analytics View */}
        {currentView === 'analytics' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className={`p-6 rounded-xl shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Word Count
                    </p>
                    <p className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                      {analytics.wordCount.toLocaleString()}
                    </p>
                    <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                      ~{Math.ceil(analytics.wordCount / 250)} pages
                    </p>
                  </div>
                  <div className="p-3 rounded-full bg-blue-100">
                    <Type className="w-6 h-6 text-blue-600" />
                  </div>
                </div>
              </div>

              <div className={`p-6 rounded-xl shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Readability Score
                    </p>
                    <p className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                      {analytics.readabilityScore}
                    </p>
                    <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                      {analytics.readabilityScore > 70 ? 'Good' : 'Needs work'}
                    </p>
                  </div>
                  <div className="p-3 rounded-full bg-green-100">
                    <Eye className="w-6 h-6 text-green-600" />
                  </div>
                </div>
              </div>

              <div className={`p-6 rounded-xl shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className={`text-sm ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                      Daily Goal
                    </p>
                    <p className={`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                      {Math.round((analytics.wordCount / settings.dailyGoal) * 100)}%
                    </p>
                    <p className={`text-xs ${isDarkMode ? 'text-gray-500' : 'text-gray-500'}`}>
                      {analytics.wordCount} / {settings.dailyGoal} words
                    </p>
                  </div>
                  <div className="p-3 rounded-full bg-purple-100">
                    <Target className="w-6 h-6 text-purple-600" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EBookEditor;