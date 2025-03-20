import json
import requests
from typing import Dict, Any, Optional, List
import urllib.parse
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

class WebSearchTool:
    """Tool for performing web searches using a search API."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine: str = "google"):
        """
        Initialize the web search tool.
        
        Args:
            api_key (str, optional): API key for the search service.
            search_engine (str): The search engine to use (google, bing, etc.)
        """
        self.name = "web_search"
        self.description = "Search the web for information. Usage: web_search <search query>"
        
        # Set up API key from parameter or environment variable
        self.api_key = api_key or os.getenv("SEARCH_API_KEY")
        if not self.api_key and search_engine in ["google", "bing"]:
            raise ValueError(f"API key must be provided for {search_engine}")
        
        self.search_engine = search_engine
    
    def __call__(self, query: str) -> str:
        """
        Execute a web search.
        
        Args:
            query (str): The search query.
            
        Returns:
            str: The search results as a formatted string.
        """
        try:
            if self.search_engine == "google":
                return self._google_search(query)
            elif self.search_engine == "duckduckgo":
                return self._ddg_search(query)
            else:
                return f"Unsupported search engine: {self.search_engine}"
        except Exception as e:
            return f"Error performing web search: {str(e)}"
    
    def _google_search(self, query: str) -> str:
        """Perform a Google search using the Custom Search JSON API."""
        # This requires a Google Custom Search Engine ID and API Key
        search_engine_id = os.getenv("GOOGLE_CSE_ID")
        if not search_engine_id:
            return "Google Custom Search Engine ID not provided"
        
        base_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": search_engine_id,
            "q": query
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        results = response.json()
        
        # Format the results
        formatted_results = "Search Results:\n\n"
        if "items" in results:
            for i, item in enumerate(results["items"][:5], 1):  # Limit to top 5 results
                formatted_results += f"{i}. {item['title']}\n"
                formatted_results += f"   URL: {item['link']}\n"
                if "snippet" in item:
                    formatted_results += f"   Snippet: {item['snippet']}\n"
                formatted_results += "\n"
        else:
            formatted_results += "No results found."
        
        return formatted_results
    
    def _ddg_search(self, query: str) -> str:
        """Perform a DuckDuckGo search using their API (no key required)."""
        # DuckDuckGo doesn't have an official API, but we can use this endpoint
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
        
        response = requests.get(url)
        response.raise_for_status()
        
        results = response.json()
        
        # Format the results
        formatted_results = "Search Results:\n\n"
        
        if "AbstractText" in results and results["AbstractText"]:
            formatted_results += f"Summary: {results['AbstractText']}\n"
            if results["AbstractURL"]:
                formatted_results += f"Source: {results['AbstractURL']}\n\n"
        
        if "RelatedTopics" in results:
            for i, topic in enumerate(results["RelatedTopics"][:5], 1):  # Limit to top 5
                if "Text" in topic:
                    formatted_results += f"{i}. {topic['Text']}\n"
                    if "FirstURL" in topic:
                        formatted_results += f"   URL: {topic['FirstURL']}\n"
                    formatted_results += "\n"
        
        if formatted_results == "Search Results:\n\n":
            formatted_results += "No results found."
        
        return formatted_results


class WeatherTool:
    """Tool for retrieving current weather information."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the weather tool.
        
        Args:
            api_key (str, optional): API key for the weather service.
        """
        self.name = "weather"
        self.description = "Get current weather for a location. Usage: weather <city name or zip code>"
        
        # Set up API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenWeather API key must be provided")
    
    def __call__(self, location: str) -> str:
        """
        Get weather information for a location.
        
        Args:
            location (str): The city name, zip code, or coordinates.
            
        Returns:
            str: The weather information as a formatted string.
        """
        try:
            # Clean up the location string
            location = location.strip()
            
            # OpenWeatherMap API endpoint
            base_url = "https://api.openweathermap.org/data/2.5/weather"
            
            # Determine if input is zip code or city name
            if location.isdigit() or (location.startswith("#") and location[1:].isdigit()):
                zip_code = location[1:] if location.startswith("#") else location
                params = {
                    "zip": f"{zip_code},us",
                    "appid": self.api_key,
                    "units": "imperial"  # Use imperial units (Fahrenheit)
                }
            else:
                params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": "imperial"  # Use imperial units (Fahrenheit)
                }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract and format weather information
            city_name = data["name"]
            country = data["sys"]["country"]
            weather_desc = data["weather"][0]["description"]
            temp = data["main"]["temp"]
            feels_like = data["main"]["feels_like"]
            humidity = data["main"]["humidity"]
            wind_speed = data["wind"]["speed"]
            
            weather_info = f"Weather for {city_name}, {country}:\n"
            weather_info += f"Conditions: {weather_desc.capitalize()}\n"
            weather_info += f"Temperature: {temp}°F (feels like {feels_like}°F)\n"
            weather_info += f"Humidity: {humidity}%\n"
            weather_info += f"Wind Speed: {wind_speed} mph\n"
            
            return weather_info
            
        except Exception as e:
            return f"Error retrieving weather information: {str(e)}"


class StockPriceTool:
    """Tool for retrieving current stock prices."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the stock price tool.
        
        Args:
            api_key (str, optional): API key for the stock data service.
        """
        self.name = "stock_price"
        self.description = "Get current stock price information. Usage: stock_price <ticker symbol>"
        
        # Set up API key from parameter or environment variable
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key must be provided")
    
    def __call__(self, ticker: str) -> str:
        """
        Get stock price information for a ticker symbol.
        
        Args:
            ticker (str): The stock ticker symbol.
            
        Returns:
            str: The stock price information as a formatted string.
        """
        try:
            # Clean up the ticker symbol
            ticker = ticker.strip().upper()
            
            # Alpha Vantage API endpoint
            base_url = "https://www.alphavantage.co/query"
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": self.api_key
            }
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check if we got a valid response
            if "Global Quote" not in data or not data["Global Quote"]:
                return f"No data found for ticker symbol {ticker}"
            
            quote = data["Global Quote"]
            
            # Extract and format stock information
            price = quote.get("05. price", "N/A")
            change = quote.get("09. change", "N/A")
            change_percent = quote.get("10. change percent", "N/A")
            high = quote.get("03. high", "N/A")
            low = quote.get("04. low", "N/A")
            volume = quote.get("06. volume", "N/A")
            
            stock_info = f"Stock Price for {ticker}:\n"
            stock_info += f"Current Price: ${price}\n"
            stock_info += f"Change: {change} ({change_percent})\n"
            stock_info += f"Today's High: ${high}\n"
            stock_info += f"Today's Low: ${low}\n"
            stock_info += f"Volume: {volume}\n"
            
            return stock_info
            
        except Exception as e:
            return f"Error retrieving stock information: {str(e)}"


class WebPageContentTool:
    """Tool for extracting content from web pages."""
    
    def __init__(self):
        """Initialize the web page content tool."""
        self.name = "webpage_content"
        self.description = "Extract text content from a webpage. Usage: webpage_content <url>"
    
    def __call__(self, url: str) -> str:
        """
        Extract text content from a web page.
        
        Args:
            url (str): The URL of the web page to extract content from.
            
        Returns:
            str: The extracted text content.
        """
        try:
            # Clean up the URL
            url = url.strip()
            
            # Add http:// if not present
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            
            # Send request with appropriate headers to avoid being blocked
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0"
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Check content type
            if "text/html" not in response.headers.get("Content-Type", ""):
                return "Error: URL does not point to an HTML page"
            
            # Use a third-party library for better HTML parsing
            try:
                from bs4 import BeautifulSoup
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style", "header", "footer", "nav"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text(separator="\n")
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                # Limit text length
                if len(text) > 2000:
                    text = text[:2000] + "...\n[Text truncated due to length]"
                
                return f"Content from {url}:\n\n{text}"
                
            except ImportError:
                # Fallback if BeautifulSoup isn't available
                # Strip HTML tags (very basic approach)
                import re
                text = re.sub(r'<[^>]*>', '', response.text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > 2000:
                    text = text[:2000] + "...\n[Text truncated due to length]"
                
                return f"Content from {url}:\n\n{text}"
            
        except Exception as e:
            return f"Error extracting content from webpage: {str(e)}"


# Usage example:
if __name__ == "__main__":
    # Test the web search tool
    search_tool = WebSearchTool(search_engine="duckduckgo")  # DuckDuckGo doesn't require an API key
    print(search_tool("What is artificial intelligence?"))
    
    # Test the weather tool if API key is available
    if os.getenv("OPENWEATHER_API_KEY"):
        weather_tool = WeatherTool()
        print(weather_tool("San Francisco"))
    
    # Test the stock price tool if API key is available
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        stock_tool = StockPriceTool()
        print(stock_tool("AAPL"))
    
    # Test the web page content tool
    webpage_tool = WebPageContentTool()
    print(webpage_tool("https://example.com")) 