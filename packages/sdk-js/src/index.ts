// packages/sdk-js/src/index.ts

export interface OptimizeOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  strategy?: 'clarity' | 'concise' | 'creative' | 'technical';
  userContext?: Record<string, any>;
}

export interface OptimizeResult {
  optimizedPrompt: string;
  strategy: string;
  model: string;
  confidence: number;
  suggestions?: string[];
}

export interface PersonalizationContext {
  userId: string;
  preferences?: Record<string, any>;
  history?: string[];
}

export class OptimizeKit {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl = 'https://api.optimizekit.dev') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  /**
   * Optimize a prompt based on strategy and context
   */
  async optimizePrompt(
    prompt: string, 
    options: OptimizeOptions = {}
  ): Promise<OptimizeResult> {
    const {
      model = 'gpt-4',
      temperature = 0.7,
      strategy = 'clarity',
      userContext = {}
    } = options;

    // Apply optimization strategies
    let optimizedPrompt = this.applyStrategy(prompt, strategy);
    
    // Add model-specific optimizations
    optimizedPrompt = this.optimizeForModel(optimizedPrompt, model);
    
    // Calculate confidence score
    const confidence = this.calculateConfidence(prompt, optimizedPrompt, strategy);

    return {
      optimizedPrompt,
      strategy,
      model,
      confidence,
      suggestions: this.generateSuggestions(prompt, strategy)
    };
  }

  /**
   * Select optimal model based on prompt characteristics
   */
  async selectModel(prompt: string, requirements?: {
    speed?: 'fast' | 'balanced' | 'quality';
    cost?: 'low' | 'medium' | 'high';
    capability?: 'basic' | 'advanced' | 'expert';
  }): Promise<{
    recommended: string;
    alternatives: string[];
    reasoning: string;
  }> {
    const promptAnalysis = this.analyzePrompt(prompt);
    const { speed = 'balanced', cost = 'medium', capability = 'advanced' } = requirements || {};

    let recommended = 'gpt-4';
    let alternatives: string[] = [];
    let reasoning = '';

    // Model selection logic
    if (promptAnalysis.complexity === 'simple' && speed === 'fast') {
      recommended = 'gpt-3.5-turbo';
      alternatives = ['claude-3-haiku', 'gemini-pro'];
      reasoning = 'Simple prompt optimized for speed and cost';
    } else if (promptAnalysis.requiresReasoning) {
      recommended = 'gpt-4';
      alternatives = ['claude-3-opus', 'gemini-pro'];
      reasoning = 'Complex reasoning task requires advanced model';
    } else if (promptAnalysis.isCreative) {
      recommended = 'claude-3-opus';
      alternatives = ['gpt-4', 'gemini-pro'];
      reasoning = 'Creative task benefits from Claude\'s writing capabilities';
    }

    return { recommended, alternatives, reasoning };
  }

  /**
   * Personalize prompt based on user context and history
   */
  async personalizePrompt(
    prompt: string,
    context: PersonalizationContext
  ): Promise<{
    personalizedPrompt: string;
    adaptations: string[];
  }> {
    const { userId, preferences = {}, history = [] } = context;
    
    let personalizedPrompt = prompt;
    const adaptations: string[] = [];

    // Apply user preferences
    if (preferences.tone) {
      personalizedPrompt = this.adjustTone(personalizedPrompt, preferences.tone);
      adaptations.push(`Adjusted tone to ${preferences.tone}`);
    }

    if (preferences.expertise) {
      personalizedPrompt = this.adjustComplexity(personalizedPrompt, preferences.expertise);
      adaptations.push(`Adjusted complexity for ${preferences.expertise} level`);
    }

    // Learn from history
    if (history.length > 0) {
      const patterns = this.extractPatterns(history);
      personalizedPrompt = this.applyLearnings(personalizedPrompt, patterns);
      adaptations.push('Applied learnings from interaction history');
    }

    return {
      personalizedPrompt,
      adaptations
    };
  }

  /**
   * Batch optimize multiple prompts
   */
  async batchOptimize(
    prompts: string[],
    options: OptimizeOptions = {}
  ): Promise<OptimizeResult[]> {
    return Promise.all(
      prompts.map(prompt => this.optimizePrompt(prompt, options))
    );
  }

  /**
   * A/B test different prompt variations
   */
  async abTest(
    variations: string[],
    testConfig: {
      metric: 'response_quality' | 'user_satisfaction' | 'task_completion';
      sampleSize: number;
    }
  ): Promise<{
    winner: string;
    results: Array<{
      prompt: string;
      score: number;
      confidence: number;
    }>;
  }> {
    // Simulate A/B test results
    const results = variations.map(prompt => ({
      prompt,
      score: Math.random() * 100,
      confidence: Math.random() * 0.5 + 0.5
    }));

    results.sort((a, b) => b.score - a.score);
    
    return {
      winner: results[0].prompt,
      results
    };
  }

  // Private helper methods
  private applyStrategy(prompt: string, strategy: string): string {
    switch (strategy) {
      case 'clarity':
        return this.enhanceClarity(prompt);
      case 'concise':
        return this.makeConcise(prompt);
      case 'creative':
        return this.enhanceCreativity(prompt);
      case 'technical':
        return this.enhanceTechnical(prompt);
      default:
        return prompt;
    }
  }

  private enhanceClarity(prompt: string): string {
    // Add structure and clear instructions
    if (!prompt.includes('Please') && !prompt.includes('Could you')) {
      prompt = `Please ${prompt.toLowerCase()}`;
    }
    
    if (!prompt.includes(':')) {
      prompt += '\n\nPlease provide a clear, step-by-step response.';
    }
    
    return prompt;
  }

  private makeConcise(prompt: string): string {
    return prompt
      .replace(/\b(please|could you|would you mind|if possible)\b/gi, '')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private enhanceCreativity(prompt: string): string {
    return `Be creative and think outside the box. ${prompt}\n\nFeel free to explore unique angles and innovative approaches.`;
  }

  private enhanceTechnical(prompt: string): string {
    return `${prompt}\n\nPlease provide technical details, specific examples, and consider edge cases in your response.`;
  }

  private optimizeForModel(prompt: string, model: string): string {
    if (model.includes('claude')) {
      return `${prompt}\n\n(Note: Please be thorough and well-structured in your response)`;
    } else if (model.includes('gpt')) {
      return prompt; // GPT models work well with most prompts as-is
    } else if (model.includes('gemini')) {
      return `Context: ${prompt}\n\nTask: Please provide a comprehensive response addressing the above.`;
    }
    return prompt;
  }

  private calculateConfidence(original: string, optimized: string, strategy: string): number {
    // Simple confidence calculation based on improvements made
    let confidence = 0.7; // Base confidence
    
    if (optimized.length > original.length * 1.1) confidence += 0.1;
    if (optimized.includes('Please') || optimized.includes('step-by-step')) confidence += 0.1;
    if (strategy === 'clarity' && optimized.includes('\n')) confidence += 0.1;
    
    return Math.min(confidence, 0.95);
  }

  private generateSuggestions(prompt: string, strategy: string): string[] {
    const suggestions: string[] = [];
    
    if (!prompt.includes('example')) {
      suggestions.push('Consider adding examples to clarify your request');
    }
    
    if (prompt.length < 20) {
      suggestions.push('Provide more context for better results');
    }
    
    if (!prompt.includes('?') && !prompt.includes('.')) {
      suggestions.push('End with a clear question or instruction');
    }
    
    return suggestions;
  }

  private analyzePrompt(prompt: string): {
    complexity: 'simple' | 'medium' | 'complex';
    requiresReasoning: boolean;
    isCreative: boolean;
    wordCount: number;
  } {
    const wordCount = prompt.split(' ').length;
    const reasoningKeywords = ['analyze', 'compare', 'evaluate', 'explain why', 'reasoning'];
    const creativeKeywords = ['creative', 'story', 'poem', 'imagine', 'brainstorm'];
    
    return {
      complexity: wordCount < 10 ? 'simple' : wordCount < 50 ? 'medium' : 'complex',
      requiresReasoning: reasoningKeywords.some(keyword => 
        prompt.toLowerCase().includes(keyword)
      ),
      isCreative: creativeKeywords.some(keyword => 
        prompt.toLowerCase().includes(keyword)
      ),
      wordCount
    };
  }

  private adjustTone(prompt: string, tone: string): string {
    switch (tone) {
      case 'formal':
        return `Please ${prompt.replace(/hey|hi/gi, '').trim()}`;
      case 'casual':
        return `Hey, ${prompt}`;
      case 'professional':
        return `I would like you to ${prompt}`;
      default:
        return prompt;
    }
  }

  private adjustComplexity(prompt: string, expertise: string): string {
    switch (expertise) {
      case 'beginner':
        return `${prompt}\n\nPlease explain in simple terms and avoid jargon.`;
      case 'expert':
        return `${prompt}\n\nFeel free to use technical terminology and advanced concepts.`;
      default:
        return prompt;
    }
  }

  private extractPatterns(history: string[]): Record<string, any> {
    // Simple pattern extraction
    const patterns = {
      averageLength: history.reduce((acc, h) => acc + h.length, 0) / history.length,
      commonWords: this.getCommonWords(history),
      tone: this.detectTone(history)
    };
    
    return patterns;
  }

  private getCommonWords(history: string[]): string[] {
    const wordCount: Record<string, number> = {};
    
    history.forEach(text => {
      text.toLowerCase().split(' ').forEach(word => {
        if (word.length > 3) {
          wordCount[word] = (wordCount[word] || 0) + 1;
        }
      });
    });
    
    return Object.entries(wordCount)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5)
      .map(([word]) => word);
  }

  private detectTone(history: string[]): string {
    const formalWords = history.join(' ').match(/\b(please|would|could|kindly)\b/gi) || [];
    const casualWords = history.join(' ').match(/\b(hey|hi|yeah|cool)\b/gi) || [];
    
    return formalWords.length > casualWords.length ? 'formal' : 'casual';
  }

  private applyLearnings(prompt: string, patterns: Record<string, any>): string {
    let enhanced = prompt;
    
    if (patterns.tone === 'formal' && !prompt.includes('please')) {
      enhanced = `Please ${enhanced}`;
    }
    
    return enhanced;
  }
}

// Export for CommonJS compatibility
export default OptimizeKit;