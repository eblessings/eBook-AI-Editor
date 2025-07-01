import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Sparkles, Check, X, Loader2, Sun, Moon, Copy, FileText,
  Bold, Italic, Underline, Link, AlignLeft, AlignCenter, 
  AlignRight, List, ListOrdered, IndentDecrease, IndentIncrease,
  Palette, MoveVertical, Upload, Download, Settings, BarChart3,
  BookOpen, Zap, Brain, Eye, Target, TrendingUp, Save,
  RefreshCw, ChevronDown, Play, Pause, Users, Globe,
  Filter, Search, Layout, Type, Image
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
  
  // Editor state
  const [showColorPicker, setShowColorPicker] = useState(false);
  const [showLinkDialog, setShowLinkDialog] = useState(false);
  const [linkUrl, setLinkUrl] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  
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
    { id: 'docx', label: 'DOCX', description: 'Microsoft Word Document', icon: FileText }
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

  // Analyze text function
  const analyzeText = useCallback(async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setIsAnalyzing(true);
    setError('');

    try {
      // Simulate AI analysis
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generate mock suggestions
      const mockSuggestions = [
        {
          category: 'grammar',
          issue: 'has changed',
          suggestion: 'have changed',
          explanation: 'Subject-verb agreement: "decades" is plural',
          confidence: 0.9
        },
        {
          category: 'style',
          issue: 'very important',
          suggestion: 'crucial',
          explanation: 'More precise word choice',
          confidence: 0.7
        },
        {
          category: 'clarity',
          issue: 'In the not-so-distant past',
          suggestion: 'Recently',
          explanation: 'Simpler, clearer expression',
          confidence: 0.8
        }
      ];

      setSuggestions(mockSuggestions);
      setAnalytics(prev => ({
        ...prev,
        readabilityScore: Math.floor(Math.random() * 30) + 60
      }));

    } catch (error) {
      setError('Analysis failed. Please try again.');
      console.error('Analysis error:', error);
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
  }, [updateContent]);

  // Dismiss suggestion
  const dismissSuggestion = useCallback((suggestion) => {
    setSuggestions(prev => prev.filter(s => s !== suggestion));
  }, []);

  // Handle file upload
  const handleFileUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (!file) return;

    setUploadProgress(10);
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadProgress(50);
      
      setTimeout(() => {
        const content = e.target.result;
        if (editorRef.current) {
          const formattedContent = content
            .split('\n\n')
            .map(paragraph => `<div style="margin-bottom: 1em;">${paragraph.replace(/\n/g, '<br>')}</div>`)
            .join('');
          
          editorRef.current.innerHTML = formattedContent;
          updateContent();
        }
        
        setUploadProgress(100);
        setTimeout(() => setUploadProgress(0), 1000);
      }, 1000);
    };
    
    reader.readAsText(file);
  }, [updateContent]);

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
    }, 3000);

    return () => clearTimeout(timeoutId);
  }, [text, projectMetadata, settings.autoSave]);

  // Load saved content
  useEffect(() => {
    const savedContent = localStorage.getItem('ebook-editor-content');
    const savedMetadata = localStorage.getItem('ebook-editor-metadata');
    
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
              <button
                onClick={() => setSettings(prev => ({ ...prev, realTimeAnalysis: !prev.realTimeAnalysis }))}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                  settings.realTimeAnalysis
                    ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                    : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                }`}
              >
                <Zap className="w-4 h-4" />
                Real-time AI
              </button>

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
            <div className={`lg:col-span-2 rounded-xl shadow-lg p-6 ${
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
                    accept=".txt,.docx,.pdf"
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
                </div>
                
                <div className="flex gap-2">
                  {!settings.realTimeAnalysis && (
                    <button
                      onClick={analyzeText}
                      disabled={isAnalyzing || !text.trim()}
                      className={`px-6 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                        isAnalyzing || !text.trim()
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
                  )}
                  
                  <button
                    onClick={() => setCurrentView('export')}
                    disabled={!text.trim()}
                    className={`px-6 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                      !text.trim()
                        ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                        : 'bg-gradient-to-r from-green-500 to-emerald-600 text-white hover:from-green-600 hover:to-emerald-700 shadow-lg'
                    }`}
                  >
                    <Download className="w-4 h-4" />
                    Export eBook
                  </button>
                </div>
              </div>

              {error && (
                <div className={`mt-4 p-3 rounded-lg ${
                  error.includes('copied') 
                    ? 'bg-green-500/10 border border-green-500/20 text-green-500'
                    : 'bg-red-500/10 border border-red-500/20 text-red-500'
                }`}>
                  <p className="text-sm">{error}</p>
                </div>
              )}
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
              </div>
            </div>
          </div>
        )}

        {/* Export View */}
        {currentView === 'export' && (
          <div className="space-y-6">
            <div className={`p-6 rounded-xl shadow-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <h3 className={`text-xl font-semibold mb-4 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                Export Format
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {exportFormats.map((format) => (
                  <div
                    key={format.id}
                    className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${
                      isDarkMode
                        ? 'border-gray-700 hover:border-gray-600 bg-gray-900'
                        : 'border-gray-300 hover:border-gray-400 bg-gray-50'
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
              
              <div className="mt-6">
                <button
                  disabled={!text.trim()}
                  className={`w-full py-3 rounded-lg font-medium transition-all flex items-center justify-center gap-2 ${
                    !text.trim()
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-purple-500 to-indigo-600 text-white hover:from-purple-600 hover:to-indigo-700 shadow-lg'
                  }`}
                >
                  <Download className="w-5 h-5" />
                  Generate & Download eBook
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default EBookEditor;