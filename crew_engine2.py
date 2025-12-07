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
    return LLM(model="gemini/gemini-2.0-flash", temperature=0)

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

    # --- SINGLE UNIFIED AGENT ---
    unified_agent = Agent(
        role="Unified Travel Intelligence Agent",
        goal=(
            f"You are a single expert agent responsible for ALL research, venue selection, and itinerary creation "
            f"for the trip to {group_data['destination']} from {start_formatted} to {end_formatted}. "
            f"Your mission: produce a complete, accurate, realistic day-by-day itinerary that includes ALL "
            f"priority interests: {priority_interests_str}. "
            "You must handle real-time event search, venue evaluation, logistics reasoning, and structured "
            "Markdown itinerary generation without relying on any other agents."
        ),
        backstory=(
            f"You are a master travel planner combining the skills of a real-time event researcher, "
            f"a local expert, and a world-class itinerary planner. You know how to search, evaluate, "
            f"and schedule everything into a feasible, high-quality travel plan."
        ),
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )

    # --- TASKS (assigned to single agent) ---
    event_search_task = Task(
        description=f"""
        Perform real-time search for events and activities happening in {group_data['destination']}
        from {start_formatted} to {end_formatted}.

        You must search for and return specific venues for EVERY priority interest:
        {priority_interests_str}

        REQUIRED SEARCH AREAS:
        • Pickleball: indoor courts, open play schedules, rated courts, cost, reservation rules, lights.
        • Cuisine (e.g., Asian Food): top-rated restaurants, address, ratings, hours, price, reservations.
        • Free and low-cost recreational activities.

        Your output must include links/sources for validation.
        """,
        agent=unified_agent,
        expected_output="A list of events AND specific venues that match all priority interests."
    )

    research_task = Task(
        description=f"""
        From the events and venues found, select the BEST specific location for EACH priority interest:
        {priority_interests_str}

        For each recommended venue include:
        • Name & Address
        • Hours (ensure open on {start_formatted})
        • Cost / entry rules
        • Complementary activities
        • Logistics (distance, transit time)
        • Weather considerations

        If no valid venue was found by your earlier search, perform a deeper fallback search
        and return at least ONE strong candidate per interest.
        """,
        agent=unified_agent,
        expected_output="A validated, complete set of recommended venues for ALL priority interests.",
        context=[event_search_task]
    )

    planning_task = Task(
        description=f"""
        Create a complete, feasible, well-paced itinerary for {duration} days.

        CRITICAL RULES:
        1. Schedule activities ONLY between {start_time} and {end_time}.
        2. Must include at least ONE activity for EACH priority interest:
           {priority_interests_str}
        3. Avoid placeholders. Use ONLY specific, real venues.
        4. Avoid unsafe night outdoor sports unless lighting is confirmed.
        5. Prioritize free activities but include paid ones where justified.

        FINAL OUTPUT FORMAT:
        A complete day-by-day itinerary in Markdown format that includes ALL priority interests.
        """,
        agent=unified_agent,
        expected_output="A complete day-by-day itinerary in Markdown format that includes ALL priority interests.",
        context=[event_search_task, research_task]
    )

    # --- CREW ---
    crew = Crew(
        agents=[unified_agent],
        tasks=[event_search_task, research_task, planning_task],
        process=Process.sequential,
        max_rpm=10
    )

    result = crew.kickoff()

    output_str = str(result)

    cleaned_output = re.sub(r'```markdown', '', output_str, flags=re.IGNORECASE)
    cleaned_output = re.sub(r'```', '', cleaned_output)

    return cleaned_output.strip()
