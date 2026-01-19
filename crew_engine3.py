from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from typing import List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import re

# Load env variables once
load_dotenv()

def get_llm():
    """Factory to get the LLM instance to ensure fresh connections."""
    return LLM(model="gemini/gemini-2.5-flash", temperature=0)

def calculate_trip_duration(start_date: str, end_date: str):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    duration = (end - start).days + 1

    dates = []
    for i in range(duration):
        day = start + timedelta(days=i)
        dates.append(day.strftime("%Y-%m-%d"))

    return duration, dates, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def aggregated_interests(people: List[Dict]) -> Dict:
    all_interests = []
    interest_count = {}

    for person in people:
        interests_list = person["interests"]

        if isinstance(interests_list, str):
            interests_list = [x.strip() for x in interests_list.split(',')]
            
        for interest in interests_list:
            clean_interest = interest.strip().title()
            all_interests.append(clean_interest)
            interest_count[clean_interest] = interest_count.get(clean_interest, 0) + 1

    sorted_by_count = sorted(interest_count.items(), key=lambda x: x[1], reverse=True)
    common_interests = [interest for interest, count in sorted_by_count if count > 1]
    unique_interests = [interest for interest, count in sorted_by_count if count == 1]

    priority_interests = common_interests + [i for i in unique_interests if i not in common_interests]

    if not priority_interests and all_interests:
        priority_interests = list(dict.fromkeys(all_interests))

    unique_list = list(dict.fromkeys(all_interests))

    return {
        "all_interests": unique_list,
        "common_interests": common_interests,
        "priority_interests": priority_interests,
        "unique_interests": unique_interests,
        "interest_summary": f"Group interests: {', '.join(unique_list)}. PRIORITY: {', '.join(priority_interests)}"
    }

def generate_itinerary(group_data: Dict) -> str:
    # Calculate dates
    duration, date_list, start_formatted, end_formatted = calculate_trip_duration(
        group_data["start_date"],
        group_data["end_date"]
    )
    
    # Time Constraints
    start_time = group_data.get('start_time', '09:00')
    end_time = group_data.get('end_time', '22:00')

    # Setup Tools
    search_tool = SerperDevTool()
    aggregate_interests = aggregated_interests(group_data['people'])
    priority_interests = aggregate_interests.get('priority_interests', aggregate_interests.get('all_interests', []))
    priority_interests_str = ", ".join(priority_interests) if priority_interests else ""
    llm = get_llm()

    # --- ONE UNIFIED AGENT ---
    unified_agent = Agent(
        role="Unified Travel Intelligence Agent",
        goal=(
            f"You are responsible for ALL research, venue selection, and itinerary creation "
            f"for the trip to {group_data['destination']} from {start_formatted} to {end_formatted}. "
            f"You must produce a complete, accurate, realistic day-by-day itinerary including ALL "
            f"priority interests: {priority_interests_str}. "
            "You must perform real-time search, evaluate venues, reason about logistics, and generate "
            "a polished Markdown itinerary."
        ),
        backstory=(
            "You combine the skills of a real-time event researcher, a local travel expert, "
            "and a world-class itinerary planner."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )

    # --- ONE UNIFIED TASK ---
    unified_task = Task(
        description=f"""
        You must complete ALL responsibilities of event research, venue selection, and itinerary planning
        in a single workflow.

        REQUIRED WORKFLOW (INTERNAL — DO NOT OUTPUT AS BULLETS):
        1. Search for events and activities happening in {group_data['destination']} between {start_formatted} and {end_formatted}.
        2. Identify specific venues for ALL priority interests: {priority_interests_str}.
            - For Pickleball: find indoor courts, open play, cost, rating, reservation rules.
            - For cuisine (e.g., Asian Food): find best restaurants with hours, price, rating, reservations.
            - Include free recreational activities and noteworthy events.
        3. Choose the BEST venue for each interest, ensuring validity for the date range.
        4. Build a feasible, realistic itinerary:
            - Activities ONLY between {start_time} and {end_time}.
            - MUST include at least one activity for every priority interest.
            - Do not schedule outdoor sports at night unless lighting is confirmed.
            - Use only specific venue names (no placeholders).
        5. Produce the final output in Markdown format.

        FINAL REQUIRED OUTPUT:
        A complete day-by-day itinerary in Markdown format that includes ALL priority interests.
        """,
        agent=unified_agent,
        expected_output="A complete day-by-day itinerary in Markdown format that includes ALL priority interests."
    )

    # --- CREW ---
    crew = Crew(
        agents=[unified_agent],
        tasks=[unified_task],
        process=Process.sequential,
        max_rpm=2
    )

    result = crew.kickoff()

    output_str = str(result)

    # Strip Markdown fences
    cleaned_output = re.sub(r'```markdown', '', output_str, flags=re.IGNORECASE)
    cleaned_output = re.sub(r'```', '', cleaned_output)

    return cleaned_output.strip()
