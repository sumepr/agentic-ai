from typing import Dict, List, Optional, Any
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import Graph, StateGraph
from langgraph.prebuilt import ToolExecutor
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SentimentAnalysisResult(BaseModel):
    """Schema for sentiment analysis results"""
    sentiment: str = Field(description="Overall sentiment (positive/negative/neutral)")
    score: float = Field(description="Sentiment score from 0-100")
    themes: List[str] = Field(description="Key themes identified in the review")
    pain_points: List[str] = Field(description="Issues or problems mentioned")
    aspects: Dict[str, str] = Field(description="Product/service aspects and their assessment")

class ReviewAnalyzer:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0,
            api_key=self.api_key
        )
        self.setup_chain()
        self.setup_graph()

    def setup_chain(self):
        """Setup the LangChain processing chain"""
        # Create prompt template
        template = """Analyze the following customer review for sentiment, themes, and specific details.

Review: {review}

Provide a detailed analysis including:
1. Overall sentiment (positive/negative/neutral)
2. Sentiment score (0-100, where 0 is most negative and 100 is most positive)
3. Key themes present in the review
4. Pain points or issues mentioned
5. Specific product/service aspects mentioned and their assessment

Respond in JSON format matching the specified schema."""

        self.prompt = ChatPromptTemplate.from_template(template)
        self.parser = JsonOutputParser(pydantic_object=SentimentAnalysisResult)

    def setup_graph(self):
        """Setup the LangGraph processing workflow"""
        workflow = StateGraph(nodes=['analyze', 'enrich', 'aggregate'])

        # Define the analysis node
        @workflow.node('analyze')
        def analyze(state):
            review = state['review']
            chain = self.prompt | self.llm | self.parser
            result = chain.invoke({"review": review})
            state['analysis'] = result
            return state

        # Define the enrichment node
        @workflow.node('enrich')
        def enrich(state):
            analysis = state['analysis']
            # Add confidence scores for themes
            themes_with_confidence = [
                {"theme": theme, "confidence": 0.8 + (0.2 * i/len(analysis.themes))}
                for i, theme in enumerate(analysis.themes)
            ]
            state['enriched_analysis'] = {
                **analysis.dict(),
                "themes_with_confidence": themes_with_confidence
            }
            return state

        # Define the aggregation node
        @workflow.node('aggregate')
        def aggregate(state):
            enriched = state['enriched_analysis']
            # Add summary metrics
            state['final_result'] = {
                **enriched,
                "summary": {
                    "overall_score": enriched['score'],
                    "theme_count": len(enriched['themes']),
                    "pain_point_count": len(enriched['pain_points'])
                }
            }
            return state

        # Connect the nodes
        workflow.add_edge('analyze', 'enrich')
        workflow.add_edge('enrich', 'aggregate')

        self.graph = workflow.compile()

    def analyze_review(self, review: str) -> Dict[str, Any]:
        """Analyze a single customer review"""
        try:
            # Initialize state with the review
            initial_state = {"review": review}
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state['final_result']
        except Exception as e:
            print(f"Error analyzing review: {e}")
            return {
                "error": str(e),
                "sentiment": "error",
                "score": 0,
                "themes": [],
                "pain_points": [],
                "aspects": {}
            }

    def analyze_reviews(self, reviews: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple reviews"""
        results = []
        for review in reviews:
            result = self.analyze_review(review)
            results.append(result)
        return results

    def get_aggregate_insights(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate aggregate insights from multiple reviews"""
        if not results:
            return {"error": "No results to analyze"}

        # Initialize counters
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        all_themes = []
        all_pain_points = []
        total_score = 0

        # Aggregate data
        for result in results:
            sentiment_counts[result['sentiment']] += 1
            total_score += result['score']
            all_themes.extend(result['themes'])
            all_pain_points.extend(result['pain_points'])

        # Calculate averages and percentages
        total_reviews = len(results)
        return {
            "total_reviews": total_reviews,
            "average_score": total_score / total_reviews,
            "sentiment_distribution": {
                k: v/total_reviews for k, v in sentiment_counts.items()
            },
            "top_themes": list(set(all_themes)),
            "common_pain_points": list(set(all_pain_points)),
            "review_quality": {
                "detailed_reviews": sum(1 for r in results if len(r['themes']) > 2),
                "brief_reviews": sum(1 for r in results if len(r['themes']) <= 2)
            }
        }

# Example usage
if __name__ == "__main__":
    # Sample reviews
    reviews = [
        "Excellent product! The quality is outstanding and customer service was very helpful. Easy to use and great value for money.",
        "Disappointed with the durability. It broke after two weeks and customer service was unresponsive. Waste of money.",
        "Average product, nothing special but gets the job done. Reasonable price point but could be better."
    ]

    # Initialize analyzer
    analyzer = ReviewAnalyzer()

    # Analyze individual reviews
    print("\nAnalyzing individual reviews...")
    results = analyzer.analyze_reviews(reviews)
    
    # Print individual results
    for i, result in enumerate(results, 1):
        print(f"\nReview {i} Analysis:")
        print(json.dumps(result, indent=2))

    # Get aggregate insights
    print("\nAggregate Insights:")
    insights = analyzer.get_aggregate_insights(results)
    print(json.dumps(insights, indent=2))