import os
import requests
import pandas as pd
import ta
import numpy as np
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import base64

# Load environment variables
load_dotenv()

class TimeframeSelector:
    def __init__(self):
        # Define timeframes up to 1 hour maximum
        self.default_timeframes = {
            'ultra_short_1': 5,     # 5 minutes
            'ultra_short_2': 15,    # 15 minutes
            'short_1': 30,          # 30 minutes
            'short_2': 60,          # 1 hour
        }
        # Lower volatility thresholds
        self.volatility_thresholds = {
            'ultra_high': 1.5,      # >1.5% price change
            'high': 1.0,            # >1.0% price change
            'medium_high': 0.75,    # >0.75% price change
            'medium': 0.5,          # >0.5% price change
            'low': 0.25             # >0.25% price change
        }
    
    def calculate_volatility(self, price_data):
        highs = price_data['high']
        lows = price_data['low']
        closes = price_data['close']
        tr1 = highs - lows
        tr2 = abs(highs - closes.shift(1))
        tr3 = abs(lows - closes.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.mean()
        current_price = closes.iloc[-1]
        volatility = (atr / current_price) * 100
        return volatility
    
    def select_optimal_timeframe(self, volatility):
        if volatility >= self.volatility_thresholds['ultra_high']:
            return self.default_timeframes['ultra_short_1']  # 5 minutes
        elif volatility >= self.volatility_thresholds['high']:
            return self.default_timeframes['ultra_short_2']  # 15 minutes
        elif volatility >= self.volatility_thresholds['medium_high']:
            return self.default_timeframes['short_1']  # 30 minutes
        else:
            return self.default_timeframes['short_2']  # 1 hour (maximum)

class CryptoBot:
    def __init__(self):
        load_dotenv()
        self.coin_mappings = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            # ... (other mappings)
        }
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.app = Application.builder().token(self.telegram_token).build()
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.geckoterminal_base_url = "https://api.geckoterminal.com/api/v2"
        self.dexscreener_base_url = "https://api.dexscreener.com/latest/dex"
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.timeframe_selector = TimeframeSelector()
        self.app.add_handler(CommandHandler("start", self.start_command))
        ticker_filter = filters.TEXT & filters.Regex(r'^\$[a-zA-Z]+')
        self.app.add_handler(MessageHandler(ticker_filter, self.handle_message))
        self.app.add_handler(MessageHandler(filters.PHOTO, self.handle_image))
    
    async def get_coin_id(self, query):
        query = query.replace('$', '').lower()
        if query in self.coin_mappings:
            return self.coin_mappings[query]
        url = f"{self.coingecko_base_url}/search"
        params = {'query': query}
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['coins']:
                return data['coins'][0]['id']
        return None
    
    async def get_market_data_from_coingecko(self, query):
        coin_id = await self.get_coin_id(query)
        if not coin_id:
            return f"Could not find {query} on CoinGecko"
        
        url = f"{self.coingecko_base_url}/coins/{coin_id}/market_chart"
        initial_params = {
            'vs_currency': 'usd',
            'days': '1'
        }
        
        initial_response = requests.get(url, headers=self.headers, params=initial_params)
        if initial_response.status_code != 200:
            return f"Error fetching price data: {initial_response.status_code}"
            
        initial_data = initial_response.json()
        prices = initial_data.get('prices', [])
        volumes = initial_data.get('total_volumes', [])
        if not prices:
            return "No price data available"
            
        times = [pd.to_datetime(p[0], unit='ms') for p in prices]
        close_prices = [p[1] for p in prices]
        temp_df = pd.DataFrame({
            'timestamp': times,
            'price': close_prices
        })
        
        ohlc = temp_df.set_index('timestamp').price.resample('15min').ohlc()
        volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        volume['timestamp'] = pd.to_datetime(volume['timestamp'], unit='ms')
        volume = volume.set_index('timestamp').resample('15min').sum()
        
        initial_df = pd.concat([ohlc, volume], axis=1)
        initial_df = initial_df.ffill()
        volatility = self.timeframe_selector.calculate_volatility(initial_df)
        optimal_minutes = self.timeframe_selector.select_optimal_timeframe(volatility)
        
        # Calculate days_needed for approximately 100 candlesticks
        days_needed = min(30, math.ceil((100 * optimal_minutes) / 1440))
        
        params = {
            'vs_currency': 'usd',
            'days': str(days_needed)
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code != 200:
            return f"Error fetching data: {response.status_code}"
            
        data = response.json()
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        times = [pd.to_datetime(p[0], unit='ms') for p in prices]
        close_prices = [p[1] for p in prices]
        temp_df = pd.DataFrame({
            'timestamp': times,
            'price': close_prices
        })
        
        ohlc = temp_df.set_index('timestamp').price.resample(f'{optimal_minutes}min').ohlc()
        volume = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
        volume['timestamp'] = pd.to_datetime(volume['timestamp'], unit='ms')
        volume = volume.set_index('timestamp').resample(f'{optimal_minutes}min').sum()
        
        df = pd.concat([ohlc, volume], axis=1)
        df = df.ffill().tail(100)  # Take the last 100 data points
        
        # Calculate technical indicators
        df['SMA5'] = ta.trend.sma_indicator(df['close'], window=5)
        df['SMA10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['SMA20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        
        # Add Stochastic RSI
        stoch_rsi = ta.momentum.StochRSIIndicator(
            close=df['close'], 
            window=14, 
            smooth1=3, 
            smooth2=3
        )
        df['StochRSI_K'] = stoch_rsi.stochrsi_k()
        df['StochRSI_D'] = stoch_rsi.stochrsi_d()
        
        # Calculate MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        market_url = f"{self.coingecko_base_url}/coins/{coin_id}"
        market_response = requests.get(market_url, headers=self.headers)
        market_data = market_response.json()
        timeframe_label = f"{optimal_minutes}m"
        
        return {
            'price': market_data['market_data']['current_price']['usd'],
            'df': df,
            'timeframe': timeframe_label,
            'volume_24h': market_data['market_data']['total_volume']['usd'],
            'change_24h': market_data['market_data']['price_change_percentage_24h'],
            'market_cap': market_data['market_data']['market_cap']['usd'],
            'name': market_data['name']
        }
    
    async def get_market_data_from_geckoterminal(self, query):
        # Check if a contract address is provided
        parts = query.split()
        if len(parts) > 1:
            # Specific coin query: $lowcap <contract_address>
            contract_address = parts[1]
            geckoterminal_url = f"{self.geckoterminal_base_url}/networks/solana/tokens/{contract_address}/pools"
        else:
            # General query: $lowcap (fetch most active low-cap coin)
            geckoterminal_url = f"{self.geckoterminal_base_url}/networks/solana/pools"
            params = {
                'page': 1,
                'include': 'base_token',
                'filter': 'volume_usd_24h.gt.1000'  # Filter for active pools
            }
        
        response = requests.get(geckoterminal_url, headers=self.headers, params=params if len(parts) == 1 else None)
        if response.status_code != 200:
            return f"Error fetching data from GeckoTerminal: {response.status_code}"
        
        data = response.json()
        pools = data.get('data', [])
        if not pools:
            return "No data available for this coin."
        
        # Extract relevant data (e.g., first pool)
        pool = pools[0]
        token_name = pool['attributes']['base_token']['name']
        token_symbol = pool['attributes']['base_token']['symbol']
        price_usd = pool['attributes']['price_usd']
        volume_24h = pool['attributes']['volume_usd']['h24']
        
        # Create a DataFrame for analysis
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [price_usd],
            'volume': [volume_24h]
        })
        
        return {
            'price': price_usd,
            'df': df,
            'timeframe': '1h',  # Default timeframe
            'volume_24h': volume_24h,
            'change_24h': 0,  # Placeholder
            'market_cap': 0,  # Placeholder
            'name': f"{token_name} ({token_symbol})"
        }
    
    async def get_market_data_from_raydium(self, query):
        # Fetch data from DEX Screener for Raydium
        dexscreener_url = f"{self.dexscreener_base_url}/pairs/solana/RAYDIUM_PAIR_ADDRESS"
        response = requests.get(dexscreener_url, headers=self.headers)
        if response.status_code != 200:
            return f"Error fetching data from DEX Screener: {response.status_code}"
        
        data = response.json()
        pair = data.get('pair', {})
        if not pair:
            return "No data available for this Raydium pair."
        
        # Extract relevant data
        token_name = pair['baseToken']['name']
        token_symbol = pair['baseToken']['symbol']
        price_usd = pair['priceUsd']
        volume_24h = pair['volume']['h24']
        
        # Create a DataFrame for analysis
        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'price': [price_usd],
            'volume': [volume_24h]
        })
        
        return {
            'price': price_usd,
            'df': df,
            'timeframe': '1h',  # Default timeframe
            'volume_24h': volume_24h,
            'change_24h': 0,  # Placeholder
            'market_cap': 0,  # Placeholder
            'name': f"{token_name} ({token_symbol})"
        }
    
    async def get_market_data(self, query):
        if query.lower().startswith("$lowcap"):
            return await self.get_market_data_from_geckoterminal(query)
        elif query.lower().startswith("$ray"):
            return await self.get_market_data_from_raydium(query)
        else:
            return await self.get_market_data_from_coingecko(query)
    
    async def create_chart(self, df, symbol, timeframe):
        # Create figure and grid
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])  # Price
        ax2 = fig.add_subplot(gs[1])  # RSI
        ax3 = fig.add_subplot(gs[2])  # Stochastic RSI
        ax4 = fig.add_subplot(gs[3])  # MACD
        
        # Style settings
        plt.style.use('default')
        bg_color = '#FFFFFF'
        grid_color = '#E5E5E5'
        spine_color = '#000000'
        
        # Configure all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor(bg_color)
            ax.grid(True, alpha=1, color=grid_color, linestyle='-', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color(spine_color)
            ax.spines['bottom'].set_color(spine_color)
            ax.spines['left'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.tick_params(axis='x', colors='black', labelsize=8)
            ax.tick_params(axis='y', colors='black', labelsize=8)
        
        # Price chart setup
        price_range = df['high'].max() - df['low'].min()
        y_min = df['low'].min() - price_range * 0.05
        y_max = df['high'].max() + price_range * 0.05
        ax1.set_ylim(y_min, y_max)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.2f}"))
        
        # Prepare x-axis data
        x = range(len(df))
        dates = [d.strftime('%b%d,%H:%M') for d in df.index]
        label_freq = max(len(df) // 10, 1)
        
        # Plot candlesticks
        for i in x:
            color = '#26a69a' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ef5350'
            body_bottom = min(df['open'].iloc[i], df['close'].iloc[i])
            body_height = abs(df['close'].iloc[i] - df['open'].iloc[i])
            ax1.add_patch(plt.Rectangle((i - 0.4, body_bottom), 0.8, body_height, color=color))
            ax1.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], color=color, linewidth=1.0)
        
        # Plot moving averages
        ax1.plot(x, df['SMA5'], label='5MA', color='orange', linewidth=1.0, alpha=1)
        ax1.plot(x, df['SMA10'], label='10MA', color='blue', linewidth=1.0, alpha=1)
        ax1.plot(x, df['SMA20'], label='20MA', color='green', linewidth=1.0, alpha=1)
        ax1.legend(fontsize=8, loc='upper right')
        
        # Plot red price line
        ax1.plot(x, df['close'], label='Price', color='red', linewidth=1.0, alpha=1)
        
        # Plot RSI
        ax2.plot(x, df['RSI'], color='#8e24aa', linewidth=1.5)
        ax2.axhline(y=70, color='#ff9999', linestyle=':', alpha=0.5)
        ax2.axhline(y=30, color='#99ff99', linestyle=':', alpha=0.5)
        ax2.fill_between(x, 70, df['RSI'].where(df['RSI'] >= 70), color='#ff9999', alpha=0.3)
        ax2.fill_between(x, 30, df['RSI'].where(df['RSI'] <= 30), color='#99ff99', alpha=0.3)
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI', fontsize=8, color='black')
        
        # Plot Stochastic RSI
        ax3.plot(x, df['StochRSI_K'], label='K', color='#2962ff', linewidth=1.0)
        ax3.plot(x, df['StochRSI_D'], label='D', color='#ff6d00', linewidth=1.0)
        ax3.axhline(y=0.8, color='#ff9999', linestyle=':', alpha=0.5)
        ax3.axhline(y=0.2, color='#99ff99', linestyle=':', alpha=0.5)
        ax3.fill_between(x, 0.8, df['StochRSI_K'].where(df['StochRSI_K'] >= 0.8), color='#ff9999', alpha=0.3)
        ax3.fill_between(x, 0.2, df['StochRSI_K'].where(df['StochRSI_K'] <= 0.2), color='#99ff99', alpha=0.3)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Stoch RSI', fontsize=8, color='black')
        ax3.legend(fontsize=8, loc='upper right')
        
        # Plot MACD
        if not df['MACD'].isnull().all() and not df['MACD_Signal'].isnull().all():
            histogram = df['MACD'] - df['MACD_Signal']
            max_val = max(
                abs(df['MACD'].max()), 
                abs(df['MACD'].min()),
                abs(df['MACD_Signal'].max()), 
                abs(df['MACD_Signal'].min())
            )
            y_range = max_val * 1.1  # Use actual MACD line values for scaling
            ax4.set_ylim(-y_range, y_range)
            ax4.plot(x, df['MACD'], label='MACD', color='#2962ff', linewidth=1.0)
            ax4.plot(x, df['MACD_Signal'], label='Signal', color='#ff6d00', linewidth=1.0)
            ax4.bar(x, histogram, color=np.where(histogram >= 0, '#26a69a80', '#ef535080'), alpha=0.3, width=0.7)
            ax4.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
            ax4.set_ylabel('MACD', fontsize=8, color='black')
            ax4.legend(fontsize=8, loc='upper right')
        
        # Hide x-axis labels for all but the bottom chart
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])
        ax3.set_xticklabels([])
        
        # Add x-axis labels only to the bottom chart
        ax4.set_xticks(range(0, len(df), label_freq))
        ax4.set_xticklabels([dates[i] for i in range(0, len(df), label_freq)], rotation=0, fontsize=8)
        
        # Capitalize the name of the pair
        title = f'{symbol.replace("$", "").upper()}/USDT · {timeframe}'
        ax1.set_title(title, loc='left', fontsize=10, pad=10, color='black')
        
        chart_path = f'analysis_{symbol.replace("$", "")}.png'
        plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor=bg_color)
        plt.close()
        return chart_path
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "Welcome! Use the bot by sending a cryptocurrency symbol, e.g., $BTC. The bot will automatically select the best timeframe based on market conditions."
        )
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message.text.strip()
        if not message.startswith('$'):
            return
        symbol = message.split()[0]
        await update.message.reply_text("Analyzing... Please wait.")
        try:
            market_data = await self.get_market_data(message)
            if isinstance(market_data, str):
                await update.message.reply_text(f"{market_data}")
                return
            df = market_data['df']
            prompt = self.generate_analysis_prompt(market_data, symbol)
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional crypto analyst. Keep responses brief and actionable."},
                    {"role": "user", "content": prompt}
                ]
            )
            chart_path = await self.create_chart(df, symbol, market_data['timeframe'])
            if chart_path:
                await update.message.reply_photo(open(chart_path, 'rb'))
                await update.message.reply_text(response.choices[0].message.content)
                os.remove(chart_path)
            else:
                await update.message.reply_text("Error generating chart")
        except Exception as e:
            await update.message.reply_text(f"Error: {str(e)}")
    
    async def handle_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Analyzing the chart... Please wait.")
        try:
            # Download the image
            photo_file = await update.message.photo[-1].get_file()
            image_path = f"chart_{update.message.from_user.id}.jpg"
            await photo_file.download_to_drive(image_path)

            # Analyze the image using OpenAI's Vision API
            analysis = await self.analyze_chart(image_path)
            await update.message.reply_text(analysis)

            # Clean up
            os.remove(image_path)
        except Exception as e:
            await update.message.reply_text(f"Error analyzing the chart: {str(e)}")
    
    async def analyze_chart(self, image_path: str) -> str:
        # Use OpenCV to preprocess the image (optional)
        image = cv2.imread(image_path)
        if image is None:
            return "Error: Unable to read the image."

        # Convert the image to base64 for OpenAI Vision API
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Define the prompt for OpenAI's Vision API
        prompt = """
        Analyze this trading chart and provide potential trade setups in the following format:

        Potential Trade Setups:

        Long Trade:
        - Entry: [Specific price or condition for entry]
        - Stop Loss: [Specific price for stop loss]
        - Take Profit: [Specific price for take profit]

        Short Trade (Counter-Trend):
        - Entry: [Specific price or condition for entry]
        - Stop Loss: [Specific price for stop loss]
        - Take Profit: [Specific price for take profit]

        Summary:
        - Provide a summary rating for both the long and short setups (e.g., Long setup: 7/10, Short setup: 4/10).
        """

        # Use OpenAI's Vision API to analyze the chart
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo",  # Updated model name
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=500,  # Increase max_tokens for detailed analysis
        )

        return response.choices[0].message.content
    
    def generate_analysis_prompt(self, market_data, symbol):
        return f"""Quick technical analysis for {market_data['name']} (${market_data['price']:.6f}):

- Trend: 
- Key Levels: 
- Trade Setup: 
- Risk: High (low market cap, potential for volatility)
- Rating: /10
"""
    
    def run(self):
        print("Bot started!")
        self.app.run_polling()

if __name__ == "__main__":
    bot = CryptoBot()
    bot.run()