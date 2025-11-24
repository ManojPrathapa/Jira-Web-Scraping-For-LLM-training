import asyncio
import random
import aiohttp
from datetime import datetime, timedelta
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.models import JiraIssue, JiraComment
from src.config import Config
from rich.console import Console

console = Console()

class JiraIngestion:
    def __init__(self):
        self.base_url = Config.JIRA_BASE_URL
        self.auth = aiohttp.BasicAuth(Config.JIRA_USERNAME, Config.JIRA_API_TOKEN)
        self.headers = {"Accept": "application/json"}

    @retry(
        stop=stop_after_attempt(5), 
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(aiohttp.ClientError)
    )
    async def _fetch_page(self, session, start_at: int, max_results: int = 50):
        """Fetches a single page with error handling."""
        url = f"{self.base_url}/rest/api/3/search"
        params = {
            "jql": f"project={Config.JIRA_PROJECT_KEY}",
            "startAt": start_at,
            "maxResults": max_results,
            "fields": "summary,description,status,priority,assignee,created,labels,comment"
        }
        async with session.get(url, params=params) as response:
            if response.status == 429:
                raise aiohttp.ClientError("Rate Limited")
            response.raise_for_status()
            return await response.json()

    async def fetch_all_issues(self) -> List[JiraIssue]:
        """Main entry point: Decides if jira account is used vs Real"""
        if Config.USE_MOCK_DATA:
            console.log(f"[yellow]⚠️ Web Scraping Mode Active: Generating {Config.MOCK_COUNT} issues...[/yellow]")
            return await self._generate_mock_data(Config.MOCK_COUNT)
        
        return await self._fetch_real_jira_data()

    async def _fetch_real_jira_data(self) -> List[JiraIssue]:
        issues = []
        start_at = 0
        total = 1 # Force entry into loop
        
        console.log("[cyan]🔄 Starting Jira API Pagination...[/cyan]")
        
        async with aiohttp.ClientSession(auth=self.auth, headers=self.headers) as session:
            while start_at < total:
                data = await self._fetch_page(session, start_at)
                
                total = data.get('total', 0)
                raw_issues = data.get('issues', [])
                
                if not raw_issues:
                    break
                
                # Transform Raw JSON -> Pydantic Models
                for raw in raw_issues:
                    fields = raw.get('fields', {})
                    
                    # Safe handling of nested fields
                    assignee = fields.get('assignee') or {}
                    priority = fields.get('priority') or {}
                    status = fields.get('status') or {}
                    
                    # Parse Comments
                    comments_raw = fields.get('comment', {}).get('comments', [])
                    parsed_comments = [
                        JiraComment(
                            author=c.get('author', {}).get('displayName', 'Unknown'),
                            body=str(c.get('body', {}))[:200], # Truncate for safety
                            created=datetime.now()
                        ) for c in comments_raw
                    ]

                    issue = JiraIssue(
                        key=raw['key'],
                        summary=fields.get('summary', ''),
                        description=fields.get('description', '') or "No description",
                        status=status.get('name', 'Unknown'),
                        priority=priority.get('name', 'Medium'),
                        assignee=assignee.get('displayName', 'Unassigned'),
                        created_at=fields.get('created', datetime.now()),
                        labels=fields.get('labels', []),
                        comments=parsed_comments,
                        url=f"{self.base_url}/browse/{raw['key']}"
                    )
                    issues.append(issue)
                
                console.log(f"   Fetched {len(issues)}/{total} issues...")
                start_at += len(raw_issues)
                
        return issues

    async def _generate_mock_data(self, count: int) -> List[JiraIssue]:
        """Generates High-Volume Synthetic Data for Stress Testing"""
        await asyncio.sleep(1)
        
        # Expanded mock data pools
        components = ["Auth Service", "Payment Gateway", "Frontend UI", "Data Pipeline", "K8s Cluster", "Redis Cache", "Search API"]
        errors = ["NullPointerException", "TimeoutError", "500 Internal Server Error", "Latency Spike", "UI Glitch", "Data Mismatch"]
        assignees = ["Alice", "Bob", "Charlie", "Diana", "Evan", "Frank", "Unassigned"]
        priorities = ["Critical", "High", "Medium", "Low"]
        statuses = ["To Do", "In Progress", "Code Review", "Done", "Won't Fix"]
        
        issues = []
        for i in range(count):
            comp = random.choice(components)
            err = random.choice(errors)
            
            # Smart description generation
            desc = f"The {comp} is failing with {err} during high load. \n\nLogs:\nError at /src/main.py:42. \nNeeds investigation."
            
            issue = JiraIssue(
                key=f"ENG-{1000+i}",
                summary=f"Fix {err} in {comp}",
                description=desc,
                status=random.choice(statuses),
                priority=random.choices(priorities, weights=[10, 20, 50, 20])[0],
                assignee=random.choice(assignees),
                created_at=datetime.now() - timedelta(days=random.randint(0, 90)),
                labels=[comp.lower().replace(" ", "-"), "bug" if "Error" in err else "task"],
                comments=[],
                url=f"https://jira.mock.com/browse/ENG-{1000+i}"
            )
            issues.append(issue)
        return issues