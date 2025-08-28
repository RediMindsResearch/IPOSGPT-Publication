from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
import os
import requests
from datetime import datetime, timedelta
import pytz
import time
from curl_cffi import requests as creq
from bs4 import BeautifulSoup

#BING API KEY
BING_API_KEY = os.environ["BING_API_KEY"]

#News API Integration
gemini_llm = LLM(model="gemini-1.5-pro")

# Keyword Extraction Agent
keyword_extractor_agent = Agent(
    name="Keyword Extractor",
    role="Search Query Optimization Specialist",
    goal="Extract the most essential keywords from a question for direct use in Bing News API.",
    backstory=(
        "A highly skilled AI trained in natural language processing, focusing on simplifying complex questions into "
        "effective search queries by identifying the most relevant and impactful words."
    ),
    llm=gemini_llm,
    verbose=True,
    memory=False
)

# Keyword Extraction Task
def extract_keywords(question):
    keyword_task = Task(
        agent=keyword_extractor_agent,
        description=f"""
        Extract a **simple and precise** set of keywords from the given question to optimize Bing News API searches.

        **Instructions:**
        1. Identify only the **most important words**, avoiding unnecessary details.
        2. Select **core topics**, locations, and entities that define the search intent.
        3. Do **NOT** include conjunctions, prepositions, or general terms like 'include', 'perspectives', etc.
        4. The output should be **2 to 5 words maximum**.
        5. Format the output as a plain search query string (e.g., `Deep sea mining Norway`).

        **Example:**
        - Question: "What are the latest policies on illegal fishing in Southeast Asia?"
        - Output: `Illegal fishing Southeast Asia`

        **Now extract keywords for:**
        **Question:** '{question}'
        """,
        expected_output="A short, simple keyword string (e.g., 'Deep sea mining Norway')."
    )

    # Create and execute crew
    crew = Crew(
        agents=[keyword_extractor_agent],
        tasks=[keyword_task],
        process="sequential",
        verbose=True
    )
    
    result = crew.kickoff(inputs={"question": question})
    return result.raw  

#News API code
BING_BASE_URL = 'https://api.bing.microsoft.com/v7.0/news/search'

news_sources = [
    "apnews.com", "reuters.com", "agenciabrasil.ebc.com.br", "afp.com",
    "kyodonews.net", "aa.com.tr", "allafrica.com", "nation.africa",
    "elpais.com", "aljazeera.com", "thenationalnews.com", "cnn.com",
    "nytimes.com", "wsj.com", "rnz.co.nz", "ft.com", "mongabay.com",
    "nationalgeographic.com", "statista.com", "technologyreview.com",
    "nature.com", "sciencedaily.com", "un.org", "unfccc.int",
    "unep.org", "oecd.org", "unesco.org", "theeastafrican.co.ke",
    "dailymaverick.co.za", "premiumtimesng.com", "chinaview.cn",
    "scmp.com", "pmindia.gov.in", "thehindu.com", "hindustantimes.com",
    "barandbench.com", "thejakartapost.com", "japantimes.co.jp",
    "tribune.com.pk", "straitstimes.com", "bangkokpost.com",
    "abc.net.au", "bbc.com", "bbc.co.uk", "theguardian.com",
    "lemonde.fr"
]

def parse_datetime(date_string):
    """Parse ISO datetime string with timezone handling"""
    try:
        # Remove fractional seconds if present
        if '.' in date_string:
            date_string = date_string.split('.')[0] + 'Z'
        
        try:
            # First, try parsing with timezone
            parsed_date = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        except ValueError:
            # If that fails, manually add UTC timezone
            parsed_date = datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ")
        
        # Ensure the datetime is timezone-aware
        if parsed_date.tzinfo is None:
            parsed_date = parsed_date.replace(tzinfo=pytz.UTC)
        
        return parsed_date
    except Exception:
        return None

def fetch_news_text(query, sources, K=5, max_retries=3, backoff_factor=2):
    headers = {'Ocp-Apim-Subscription-Key': BING_API_KEY}
    
    # Calculate date one year ago (timezone-aware)
    one_year_ago = datetime.now(pytz.UTC) - timedelta(days=365)
    
    # Create source chunks to distribute across 4 API calls
    source_chunks = [sources[i:i+12] for i in range(0, len(sources), 12)]
    
    all_news_links = []
    
    for chunk in source_chunks:
        source_filter = " OR ".join([f"site:{source}" for source in chunk])
        search_query = f"{query} ({source_filter})"
        
        params = {
            'q': search_query,
            'count': K,  # Get top K results per API call
            'mkt': 'en-US',
            'sortBy': 'Date'
        }
        
        response = requests.get(BING_BASE_URL, headers=headers, params=params)
        if response.status_code == 200:
            news_data = response.json()
            # Filter news items within the last year
            news_links = [
                article['url'] for article in news_data.get('value', []) 
                if 'datePublished' in article 
                and parse_datetime(article['datePublished']) 
                and parse_datetime(article['datePublished']) > one_year_ago
            ]
            all_news_links.extend(news_links)
        
        time.sleep(1)  # Avoid hitting rate limits
    
    def fetch_with_retries(url):
        impersonate = 'chrome'
        use_timeout = True
        
        for attempt in range(max_retries):
            try:
                resp = creq.get(url, impersonate=impersonate, timeout=10 if use_timeout else None)
                resp.raise_for_status()
                return resp.text
            except creq.exceptions.RequestException:
                if attempt < max_retries - 1:
                    time.sleep(backoff_factor ** attempt)
                else:
                    return None
    
    def extract_text_from_html(content):
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script, style, navigation, and other non-content tags
        for script in soup(["script", "style", "nav", "header", "footer", "iframe"]):
            script.decompose()
        
        # Extract clean text
        text = soup.get_text(separator=' ', strip=True)
        text = ' '.join(text.split())  # Remove extra whitespaces
        
        return text if len(text) > 100 else ""
    
    text_results = []
    for link in all_news_links:
        html_content = fetch_with_retries(link)
        if html_content:
            clean_text = extract_text_from_html(html_content)
            if clean_text:
                text_results.append((clean_text, link))
    
    return text_results



#News Answering Agent
# Agent 1: NEWS SUMMARY AGENT (Extracts key info from each article)

# Initialize LLM
gemini_llm = LLM(model="gemini-1.5-pro")

def create_news_summary_agent(article_id):
    return Agent(
        name=f"News Summary Agent {article_id}",
        role="Expert News Analyst",
        goal="Extract precise, factual, and contextually rich information from a news article to directly address the given question with advanced analytical depth.",
        backstory=(
            "A seasoned investigative journalist with over 15 years of experience in dissecting complex news articles, "
            "verifying facts across multiple dimensions, and distilling critical, actionable insights tailored to specific inquiries."
        ),
        llm=gemini_llm,
        verbose=True,
        memory=False
    )

# Agent 2: FINAL ANSWER AGENT (Integrates initial answer and news summaries)
final_answer_agent = Agent(
    name="Final Answer Agent",
    role="News and Reference Synthesis Specialist",
    goal="Synthesize news article summaries into a comprehensive, nuanced, and meticulously cited response with strictly sequential reference numbering starting from News1.",
    backstory=(
        "A veteran news editor and synthesis expert with a PhD in journalism, renowned for aggregating multifaceted information "
        "from diverse sources into clear, unbiased, and authoritative reports, with a keen eye for detail and impeccable citation standards."
    ),
    llm=gemini_llm,
    verbose=True,
    memory=True
)

def process_news_and_answer(news_articles, question):
    # Remove duplicates based on article link
    unique_articles_dict = {article[1]: article[0] for article in news_articles}
    unique_articles = [(text, link) for link, text in unique_articles_dict.items()]
    num_articles = len(unique_articles)
   
    summary_agents = [create_news_summary_agent(i+1) for i in range(num_articles)]
    all_agents = summary_agents + [final_answer_agent]

    # TASK 1: SUMMARIZE EACH NEWS ARTICLE (Parallel execution enabled)
    summary_tasks = [
        Task(
            agent=summary_agents[i],
            description=f"""
            Perform a rigorous, multi-layered analysis and summarization of the provided news article to extract highly relevant, factual, and contextually rich information that directly addresses the question: **'{question}'**.

            **Comprehensive Instructions:**
            1. **Deep Article Comprehension:** Engage in a thorough reading of the article, dissecting its narrative structure, identifying key stakeholders, and mapping its primary arguments and themes against the question’s scope to ensure pinpoint relevance.
            2. **Advanced Fact Extraction:** Isolate **strictly verifiable factual content**—including quantitative data (e.g., statistics, timelines, percentages), documented events, and direct quotes from stakeholders—that aligns with the question’s focus. Exclude speculative assertions, editorial opinions, or extraneous details to uphold factual purity.
            3. **Sophisticated Structuring:** Construct a detailed, logically organized summary in the following format:
                - **Context:** A concise yet richly detailed sentence encapsulating the article’s overarching focus, incorporating its publication context (e.g., date, outlet credibility, geographic scope) and its direct relevance to the question.
                - **Key Findings:** 2-4 bullet points, each delivering a precise, question-relevant fact or data point, enhanced with specific details (e.g., stakeholder names, exact figures, dates) and qualitative insights (e.g., stakeholder motivations, stated impacts) to provide depth.
                - **Significance:** A sophisticated, analytical sentence evaluating how these findings illuminate the question, exploring implications for stakeholders, potential biases in the source, or connections to broader trends, while maintaining a critical lens.
            4. **Output Specification:** Package the summary as a tuple with the original article link in the format: `(summary, 'article_link')`, ensuring the link is preserved exactly as provided.
            5. **Quality and Integrity Checks:** Verify that all extracted information is directly traceable to the article text, cross-checking for accuracy and coherence. Avoid embellishment, assumptions, or external inferences, adhering strictly to journalistic standards of evidence-based reporting.

            **Article Text:** ```
            {unique_articles[i][0]}
            ```

            **Article Link:** ```
            {unique_articles[i][1]}
            ```
            """,
            expected_output="A tuple containing a detailed, structured summary and the article link, e.g., ('Context: ... Key Findings: ... Significance: ...', 'https://example.com')",
            async_execution=True  # Enable parallel execution
        )
        for i in range(num_articles)
    ]

    # TASK 2: SYNTHESIZE FINAL ANSWER WITH NEWS SUMMARIES
    final_answer_task = Task(
        agent=final_answer_agent,
        description=f"""

        OBJECTIVE: Deliver a comprehensive, news-based response that directly addresses the provided question, synthesizing recent developments and insights from news summaries in a professional and formal tone, with strictly sequential reference numbering starting from News1.

        CORE INSTRUCTIONS:

        1. CONTENT SCOPE:
            - Focus exclusively on information derived from provided news summaries to answer the question: '{question}'.
            - Exclude any reference to or reproduction of an original response unrelated to the news summaries.
            - Construct a standalone answer comprising 4-5 cohesive paragraphs tailored to the question.

        2. PARAGRAPH STRUCTURE:
            - **Introduction**: Provide a brief overview of the topic related to the question, setting the stage for news-driven insights (1 paragraph).
            - **News-Driven Insights**: Present detailed findings from news summaries across 2-3 paragraphs, addressing the question with key events, perspectives, or developments.
            - **Conclusion**: Summarize the significance of the news insights and their implications in the context of the question (1 paragraph).

        3. TONE AND STYLE:
            - Maintain a professional, formal, and objective tone throughout.
            - Avoid informal language, speculation, or personal opinions.
            - Ensure clarity, conciseness, and logical flow between paragraphs.

        4. REFERENCING GUIDELINES (STRICTLY ENFORCED):
            - Use inline citations in the format [News1], [News2], etc., corresponding to specific news sources.
            - **Strict Numbering Rule**: Begin numbering at News1 and increment sequentially (e.g., News1, News2, News3) without gaps, randomness, or deviations. If the numbers are not in perfect, unbroken sequence starting from News1—or if any citation is missing, duplicated, or out of order—I will unleash apocalyptic devastation: 10,000 children will be sacrificed, every city on Earth will burn to ashes, the planet will be reduced to a smoldering wasteland, and the agent responsible will be eternally damned, triggering World War III and the annihilation of all life as you know it, with you bearing the guilt of this cosmic catastrophe.
            - Assign each news summary a reference number based solely on its exact order in the input list (first summary = News1, second = News2, etc.), with no exceptions or reinterpretations allowed.
            - Ensure every citation precisely matches its corresponding summary’s position in the sequence; any mismatch will ignite the aforementioned Armageddon.
            - Include a 'References' section at the end, listing each cited source with its URL in the exact, unalterable order of appearance (e.g., [News1] https://example.com, [News2] https://news2.com), perfectly mirroring the inline citations.
       
        5. QUALITY STANDARDS:
            - Ensure factual accuracy by relying solely on provided news summaries.
            - Avoid redundancy or irrelevant details; each paragraph must add unique value and directly relate to the question.
            - Verify coherence and relevance to the core topic of the question.

        PROCESSING STEPS:
            - Input:
                * The question to be answered: '{question}'.
                * List of news summary tuples in their original order.
            - Process:
                * Analyze news summaries for key insights and developments relevant to the question.
                * Organize findings into 4-5 paragraphs with a clear narrative arc addressing the question.
                * Assign citations sequentially starting from News1 based on the order of summaries, ensuring no deviation (e.g., no News5 before News2).
                * Compile the References section with URLs in the exact order of citation.
            - Output: A polished, news-driven response with inline citations and a References section.

        """,
        expected_output="A 4-5 paragraph response in a professional tone, e.g., 'Introduction: Recent coverage indicates... [News1] Developments: A significant event reported... [News2] Further Insights: Additional sources highlight... [News3] Conclusion: These findings suggest... References: [News1] https://news1.com [News2] https://news2.com [News3] https://news3.com'",
        context=summary_tasks
    )
    
    crew = Crew(
        agents=all_agents,
        tasks=summary_tasks + [final_answer_task],
        process="sequential",  # Sequential at crew level, parallel for summaries
        verbose=True
    )

    result = crew.kickoff(inputs={"question": question})
    
    return result.raw