# packages/sdk-py/optimizekit/__init__.py

"""
OptimizeKit Python SDK

A comprehensive toolkit for AI prompt optimization, model selection, and personalization.
"""

import asyncio
import re
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import json


class Strategy(Enum):
    CLARITY = "clarity"
    CONCISE = "concise"
    CREATIVE = "creative"
    TECHNICAL = "technical"


class Complexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


@dataclass
class OptimizeResult:
    """Result of prompt optimization"""
    optimized_prompt: str
    strategy: str
    model: str
    confidence: float
    suggestions: List[str]
    original_length: int
    optimized_length: int
    improvement_score: float = 0.0


@dataclass
class ModelRecommendation:
    """Model selection recommendation"""
    recommended: str
    alternatives: List[str]
    reasoning: str
    cost_estimate: float
    speed_rating: str
    capability_score: float


@dataclass
class PersonalizationResult:
    """Result of prompt personalization"""
    personalized_prompt: str
    adaptations: List[str]
    confidence: float
    user_profile_updates: Dict[str, Any]


@dataclass
class ABTestResult:
    """A/B test results"""
    winner: str
    results: List[Dict[str, Union[str, float]]]
    statistical_significance: float
    confidence_interval: Tuple[float, float]


class OptimizeKit:
    """
    OptimizeKit Python SDK
    
    Provides AI prompt optimization, model selection, and personalization capabilities.
    """
    
    def __init__(self, api_key: str, base_url: str = "https://api.optimizekit.dev"):
        """
        Initialize OptimizeKit client
        
        Args:
            api_key: Your OptimizeKit API key
            base_url: API base URL (optional)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'OptimizeKit-Python/1.0.0'
        })
    
    def optimize_prompt(
        self,
        prompt: str,
        strategy: Union[Strategy, str] = Strategy.CLARITY,
        model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> OptimizeResult:
        """
        Optimize a prompt using specified strategy
        
        Args:
            prompt: Original prompt to optimize
            strategy: Optimization strategy to use
            model: Target model for optimization
            temperature: Model temperature setting
            max_tokens: Maximum tokens for the model
            user_context: Additional user context
            
        Returns:
            OptimizeResult with optimized prompt and metadata
        """
        if isinstance(strategy, Strategy):
            strategy = strategy.value
            
        # Apply optimization strategy
        optimized_prompt = self._apply_strategy(prompt, strategy)
        
        # Apply model-specific optimizations
        optimized_prompt = self._optimize_for_model(optimized_prompt, model)
        
        # Calculate metrics
        confidence = self._calculate_confidence(prompt, optimized_prompt, strategy)
        suggestions = self._generate_suggestions(prompt, strategy)
        improvement_score = self._calculate_improvement_score(prompt, optimized_prompt)
        
        return OptimizeResult(
            optimized_prompt=optimized_prompt,
            strategy=strategy,
            model=model,
            confidence=confidence,
            suggestions=suggestions,
            original_length=len(prompt),
            optimized_length=len(optimized_prompt),
            improvement_score=improvement_score
        )
    
    def select_model(
        self,
        prompt: str,
        speed_preference: str = "balanced",  # "fast", "balanced", "quality"
        cost_preference: str = "medium",     # "low", "medium", "high"
        capability_requirement: str = "advanced"  # "basic", "advanced", "expert"
    ) -> ModelRecommendation:
        """
        Select optimal model based on prompt characteristics and requirements
        
        Args:
            prompt: The prompt to analyze
            speed_preference: Speed requirement preference
            cost_preference: Cost tolerance
            capability_requirement: Required capability level
            
        Returns:
            ModelRecommendation with suggested model and alternatives
        """
        analysis = self._analyze_prompt(prompt)
        
        # Model selection logic
        if analysis['complexity'] == Complexity.SIMPLE and speed_preference == "fast":
            recommended = "gpt-3.5-turbo"
            alternatives = ["claude-3-haiku", "gemini-pro"]
            reasoning = "Simple prompt optimized for speed and cost efficiency"
            cost_estimate = 0.002
            speed_rating = "fast"
            capability_score = 0.8
        elif analysis['requires_reasoning']:
            recommended = "gpt-4"
            alternatives = ["claude-3-opus", "gemini-pro"]
            reasoning = "Complex reasoning task requires advanced model capabilities"
            cost_estimate = 0.03
            speed_rating = "medium"
            capability_score = 0.95
        elif analysis['is_creative']:
            recommended = "claude-3-opus"
            alternatives = ["gpt-4", "gemini-pro"]
            reasoning = "Creative task benefits from Claude's superior writing capabilities"
            cost_estimate = 0.015
            speed_rating = "medium"
            capability_score = 0.92
        elif cost_preference == "low":
            recommended = "gpt-3.5-turbo"
            alternatives = ["claude-3-haiku"]
            reasoning = "Cost-optimized selection for budget-conscious usage"
            cost_estimate = 0.002
            speed_rating = "fast"
            capability_score = 0.75
        else:
            recommended = "gpt-4"
            alternatives = ["claude-3-opus", "gemini-pro"]
            reasoning = "Balanced choice for general-purpose tasks"
            cost_estimate = 0.03
            speed_rating = "medium"
            capability_score = 0.9
        
        return ModelRecommendation(
            recommended=recommended,
            alternatives=alternatives,
            reasoning=reasoning,
            cost_estimate=cost_estimate,
            speed_rating=speed_rating,
            capability_score=capability_score
        )
    
    def personalize_prompt(
        self,
        prompt: str,
        user_id: str,
        preferences: Optional[Dict[str, Any]] = None,
        history: Optional[List[str]] = None
    ) -> PersonalizationResult:
        """
        Personalize prompt based on user context and history
        
        Args:
            prompt: Original prompt
            user_id: Unique user identifier
            preferences: User preferences dictionary
            history: List of previous prompts/interactions
            
        Returns:
            PersonalizationResult with personalized prompt
        """
        preferences = preferences or {}
        history = history or []
        
        personalized_prompt = prompt
        adaptations = []
        
        # Apply tone preferences
        if 'tone' in preferences:
            personalized_prompt = self._adjust_tone(personalized_prompt, preferences['tone'])
            adaptations.append(f"Adjusted tone to {preferences['tone']}")
        
        # Apply expertise level
        if 'expertise' in preferences:
            personalized_prompt = self._adjust_complexity(
                personalized_prompt, 
                preferences['expertise']
            )
            adaptations.append(f"Adjusted complexity for {preferences['expertise']} level")
        
        # Apply learning from history
        if history:
            patterns = self._extract_patterns(history)
            personalized_prompt = self._apply_learnings(personalized_prompt, patterns)
            adaptations.append("Applied learnings from interaction history")
        
        # Apply language preferences
        if 'language_style' in preferences:
            personalized_prompt = self._adjust_language_style(
                personalized_prompt, 
                preferences['language_style']
            )
            adaptations.append(f"Adjusted to {preferences['language_style']} language style")
        
        confidence = min(0.9, 0.6 + len(adaptations) * 0.1)
        
        # Generate profile updates
        profile_updates = {
            'last_optimization': prompt,
            'preferred_adaptations': adaptations,
            'interaction_count': len(history) + 1
        }
        
        return PersonalizationResult(
            personalized_prompt=personalized_prompt,
            adaptations=adaptations,
            confidence=confidence,
            user_profile_updates=profile_updates
        )
    
    def batch_optimize(
        self,
        prompts: List[str],
        strategy: Union[Strategy, str] = Strategy.CLARITY,
        model: str = "gpt-4",
        **kwargs
    ) -> List[OptimizeResult]:
        """
        Optimize multiple prompts in batch
        
        Args:
            prompts: List of prompts to optimize
            strategy: Optimization strategy
            model: Target model
            **kwargs: Additional optimization parameters
            
        Returns:
            List of OptimizeResult objects
        """
        return [
            self.optimize_prompt(prompt, strategy, model, **kwargs)
            for prompt in prompts
        ]
    
    def ab_test(
        self,
        variations: List[str],
        test_config: Dict[str, Any]
    ) -> ABTestResult:
        """
        Perform A/B testing on prompt variations
        
        Args:
            variations: List of prompt variations to test
            test_config: Test configuration including metric and sample_size
            
        Returns:
            ABTestResult with winner and detailed results
        """
        metric = test_config.get('metric', 'response_quality')
        sample_size = test_config.get('sample_size', 100)
        
        # Simulate A/B test results (in real implementation, this would use actual data)
        import random
        
        results = []
        for i, variation in enumerate(variations):
            # Simulate performance metrics
            base_score = random.uniform(70, 95)
            noise = random.gauss(0, 5)
            score = max(0, min(100, base_score + noise))
            
            results.append({
                'prompt': variation,
                'score': score,
                'confidence': random.uniform(0.7, 0.95),
                'sample_size': sample_size,
                'conversion_rate': score / 100,
                'p_value': random.uniform(0.01, 0.1)
            })
        
        # Sort by score to find winner
        results.sort(key=lambda x: x['score'], reverse=True)
        winner = results[0]['prompt']
        
        # Calculate statistical significance
        statistical_significance = 1 - results[0]['p_value']
        confidence_interval = (
            results[0]['score'] - 5,
            results[0]['score'] + 5
        )
        
        return ABTestResult(
            winner=winner,
            results=results,
            statistical_significance=statistical_significance,
            confidence_interval=confidence_interval
        )
    
    def get_analytics(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get usage analytics and insights
        
        Args:
            start_date: Start date for analytics (YYYY-MM-DD)
            end_date: End date for analytics (YYYY-MM-DD)
            user_id: Filter by specific user
            
        Returns:
            Analytics data dictionary
        """
        # Simulate analytics data
        import random
        from datetime import datetime, timedelta
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        return {
            'period': {'start': start_date, 'end': end_date},
            'total_requests': random.randint(1000, 10000),
            'optimization_rate': round(random.uniform(85, 98), 1),
            'avg_improvement': round(random.uniform(15, 35), 1),
            'most_used_strategy': random.choice(['clarity', 'creative', 'technical']),
            'model_distribution': {
                'gpt-4': random.randint(40, 60),
                'claude-3-opus': random.randint(20, 35),
                'gemini-pro': random.randint(10, 25)
            },
            'cost_savings': round(random.uniform(200, 1000), 2),
            'user_satisfaction': round(random.uniform(4.2, 4.9), 1)
        }
    
    # Private helper methods
    def _apply_strategy(self, prompt: str, strategy: str) -> str:
        """Apply optimization strategy to prompt"""
        strategies = {
            'clarity': self._enhance_clarity,
            'concise': self._make_concise,
            'creative': self._enhance_creativity,
            'technical': self._enhance_technical
        }
        
        if strategy in strategies:
            return strategies[strategy](prompt)
        return prompt
    
    def _enhance_clarity(self, prompt: str) -> str:
        """Enhance prompt clarity"""
        enhanced = prompt.strip()
        
        # Add polite phrasing if missing
        if not any(word in enhanced.lower() for word in ['please', 'could you', 'would you']):
            enhanced = f"Please {enhanced.lower()}" if enhanced else enhanced
        
        # Add structure request
        if not any(phrase in enhanced.lower() for phrase in ['step-by-step', 'clearly', 'detailed']):
            enhanced += "\n\nPlease provide a clear, step-by-step response."
        
        # Ensure proper punctuation
        if enhanced and not enhanced.endswith(('.', '?', '!')):
            enhanced += '.'
        
        return enhanced
    
    def _make_concise(self, prompt: str) -> str:
        """Make prompt more concise"""
        # Remove unnecessary politeness
        concise = re.sub(r'\b(please|could you|would you mind|if possible)\b', '', prompt, flags=re.IGNORECASE)
        
        # Remove redundant phrases
        concise = re.sub(r'\b(i would like|i want|i need)\b', '', concise, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        concise = re.sub(r'\s+', ' ', concise).strip()
        
        return concise
    
    def _enhance_creativity(self, prompt: str) -> str:
        """Enhance prompt for creative tasks"""
        creative_prefix = "Be creative and think outside the box. "
        creative_suffix = "\n\nFeel free to explore unique angles and innovative approaches."
        
        if not prompt.lower().startswith('be creative'):
            prompt = creative_prefix + prompt
        
        if 'creative' not in prompt.lower() and 'innovative' not in prompt.lower():
            prompt += creative_suffix
        
        return prompt
    
    def _enhance_technical(self, prompt: str) -> str:
        """Enhance prompt for technical tasks"""
        technical_suffix = "\n\nPlease provide technical details, specific examples, and consider edge cases in your response."
        
        if 'technical' not in prompt.lower() and 'detailed' not in prompt.lower():
            prompt += technical_suffix
        
        return prompt
    
    def _optimize_for_model(self, prompt: str, model: str) -> str:
        """Apply model-specific optimizations"""
        if 'claude' in model.lower():
            return f"{prompt}\n\n(Please be thorough and well-structured in your response)"
        elif 'gpt' in model.lower():
            return prompt  # GPT models work well with most prompts as-is
        elif 'gemini' in model.lower():
            return f"Context: {prompt}\n\nTask: Please provide a comprehensive response addressing the above."
        
        return prompt
    
    def _analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt characteristics"""
        word_count = len(prompt.split())
        
        reasoning_keywords = ['analyze', 'compare', 'evaluate', 'explain why', 'reasoning', 'because']
        creative_keywords = ['creative', 'story', 'poem', 'imagine', 'brainstorm', 'innovative']
        
        complexity = Complexity.SIMPLE
        if word_count > 50:
            complexity = Complexity.COMPLEX
        elif word_count > 15:
            complexity = Complexity.MEDIUM
        
        return {
            'complexity': complexity,
            'word_count': word_count,
            'requires_reasoning': any(keyword in prompt.lower() for keyword in reasoning_keywords),
            'is_creative': any(keyword in prompt.lower() for keyword in creative_keywords),
            'has_examples': 'example' in prompt.lower(),
            'has_constraints': any(word in prompt.lower() for word in ['must', 'should', 'limit', 'within'])
        }
    
    def _calculate_confidence(self, original: str, optimized: str, strategy: str) -> float:
        """Calculate optimization confidence score"""
        base_confidence = 0.7
        
        # Length improvement
        if len(optimized) > len(original) * 1.1:
            base_confidence += 0.1
        
        # Structure improvements
        if any(phrase in optimized.lower() for phrase in ['please', 'step-by-step', 'clearly']):
            base_confidence += 0.1
        
        # Strategy-specific bonuses
        if strategy == 'clarity' and '\n' in optimized:
            base_confidence += 0.05
        
        return min(base_confidence, 0.95)
    
    def _calculate_improvement_score(self, original: str, optimized: str) -> float:
        """Calculate improvement score as percentage"""
        # Simple heuristic based on length and structure improvements
        base_score = 0
        
        if len(optimized) > len(original):
            base_score += 10
        
        if optimized.count('\n') > original.count('\n'):
            base_score += 15
        
        if any(word in optimized.lower() for word in ['please', 'clear', 'specific']):
            base_score += 20
        
        return min(base_score, 50)
    
    def _generate_suggestions(self, prompt: str, strategy: str) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if 'example' not in prompt.lower():
            suggestions.append('Consider adding examples to clarify your request')
        
        if len(prompt) < 20:
            suggestions.append('Provide more context for better results')
        
        if not prompt.strip().endswith(('?', '.', '!')):
            suggestions.append('End with a clear question or instruction')
        
        if strategy == 'technical' and 'specific' not in prompt.lower():
            suggestions.append('Add specific technical requirements or constraints')
        
        return suggestions
    
    def _adjust_tone(self, prompt: str, tone: str) -> str:
        """Adjust prompt tone"""
        if tone == 'formal':
            return f"Please {prompt.replace('hey', '').replace('hi', '').strip()}"
        elif tone == 'casual':
            return f"Hey, {prompt}"
        elif tone == 'professional':
            return f"I would like you to {prompt}"
        
        return prompt
    
    def _adjust_complexity(self, prompt: str, expertise: str) -> str:
        """Adjust prompt complexity level"""
        if expertise == 'beginner':
            return f"{prompt}\n\nPlease explain in simple terms and avoid technical jargon."
        elif expertise == 'expert':
            return f"{prompt}\n\nFeel free to use technical terminology and advanced concepts."
        
        return prompt
    
    def _adjust_language_style(self, prompt: str, style: str) -> str:
        """Adjust language style"""
        if style == 'academic':
            return f"{prompt}\n\nPlease respond in an academic style with proper citations where relevant."
        elif style == 'conversational':
            return f"{prompt}\n\nPlease respond in a conversational, friendly manner."
        elif style == 'business':
            return f"{prompt}\n\nPlease provide a business-focused response with actionable insights."
        
        return prompt
    
    def _extract_patterns(self, history: List[str]) -> Dict[str, Any]:
        """Extract patterns from user history"""
        if not history:
            return {}
        
        avg_length = sum(len(h) for h in history) / len(history)
        
        # Count common words
        word_counts = {}
        for text in history:
            for word in text.lower().split():
                if len(word) > 3:
                    word_counts[word] = word_counts.get(word, 0) + 1
        
        common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Detect tone
        formal_indicators = sum(text.lower().count(word) for text in history for word in ['please', 'would', 'could'])
        casual_indicators = sum(text.lower().count(word) for text in history for word in ['hey', 'hi', 'yeah'])
        
        return {
            'avg_length': avg_length,
            'common_words': [word for word, _ in common_words],
            'tone': 'formal' if formal_indicators > casual_indicators else 'casual',
            'total_interactions': len(history)
        }
    
    def _apply_learnings(self, prompt: str, patterns: Dict[str, Any]) -> str:
        """Apply learned patterns to prompt"""
        enhanced = prompt
        
        # Apply tone learning
        if patterns.get('tone') == 'formal' and 'please' not in prompt.lower():
            enhanced = f"Please {enhanced}"
        
        # Apply length preference
        avg_length = patterns.get('avg_length', 0)
        if avg_length > 50 and len(prompt) < 30:
            enhanced += "\n\nPlease provide a detailed and comprehensive response."
        
        return enhanced


# Async version for advanced users
class AsyncOptimizeKit(OptimizeKit):
    """Async version of OptimizeKit for concurrent operations"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.optimizekit.dev"):
        super().__init__(api_key, base_url)
    
    async def optimize_prompt_async(self, *args, **kwargs) -> OptimizeResult:
        """Async version of optimize_prompt"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.optimize_prompt, *args, **kwargs)
    
    async def batch_optimize_async(self, prompts: List[str], **kwargs) -> List[OptimizeResult]:
        """Async batch optimization"""
        tasks = [self.optimize_prompt_async(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)


# Export main classes
__all__ = [
    'OptimizeKit',
    'AsyncOptimizeKit',
    'Strategy',
    'Complexity',
    'OptimizeResult',
    'ModelRecommendation',
    'PersonalizationResult',
    'ABTestResult'
]