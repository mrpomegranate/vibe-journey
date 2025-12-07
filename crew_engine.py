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
    """
    Aggregates interests, ensuring case-insensitivity so 'Pickleball'
    and 'pickleball' match.
    """
    all_interests = []
    interest_count = {}

    for person in people:
        interests_list = person["interests"]

        # Handle string input if comma-separated
        if isinstance(interests_list, str):
            interests_list = [x.strip() for x in interests_list.split(',')]
            
        for interest in interests_list:
            # Normalize to title case for consistency (e.g. "pickleball" -> "Pickleball")
            clean_interest = interest.strip().title()
            all_interests.append(clean_interest)
            interest_count[clean_interest] = interest_count.get(clean_interest, 0) + 1

    sorted_by_count = sorted(interest_count.items(), key=lambda x: x[1], reverse=True)
    common_interests = [interest for interest, count in sorted_by_count if count > 1]
    unique_interests = [interest for interest, count in sorted_by_count if count == 1]

    # Build priority list: common first, then unique (preserve order)
    priority_interests = common_interests + [i for i in unique_interests if i not in common_interests]

    # If there are no repeated interests, still keep all unique interests as priorities
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
    """
    Main entry point called by the Flask App.
    """
    
    # Calculate dates
    duration, date_list, start_formatted, end_formatted = calculate_trip_duration(
        group_data["start_date"],
        group_data["end_date"]
    )
    
    # Time Constraints
    start_time = group_data.get('start_time', '09:00')
    end_time = group_data.get('end_time', '22:00')

    # Setup Tools & Analytics
    search_tool = SerperDevTool()
    aggregate_interests = aggregated_interests(group_data['people'])
    priority_interests = aggregate_interests.get('priority_interests', aggregate_interests.get('all_interests', []))
    priority_interests_str = ", ".join(priority_interests) if priority_interests else ""
    # common_interests_str = ", ".join(aggregate_interests['common_interests'])
    llm = get_llm()

    # --- AGENTS ---
    event_researcher = Agent(
        role="Real-Time Event and Activity Researcher",
        # 🔧 UPDATED: stronger and more specific search instructions
        goal=(
            f"Find real-time events, activities, and venues in {group_data['destination']} "
            f"for the travel dates that match these PRIORITY interests: {priority_interests_str}. "
            "For sports (e.g., Pickleball), prioritize: 'indoor' locations, 'open play' schedules, "
            "'rated' or 'recommended' courts, and any facilities with reservation policies. "
            "For cuisine interests (e.g., 'Asian Food'), search for top-rated restaurants and "
            "sub-cuisines (Chinese, Japanese, Korean, Vietnamese) and gather ratings, price range, and opening hours."
        ),
        backstory=f"""You are an expert at finding current events, up-to-date local listings,
        and venue-specific logistics (hours, cost, open-play times). You search for indoor
        sports facilities, community rec centers, and top-rated restaurants during the trip dates.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
)


    local_expert = Agent(
        role="Local Travel Expert",
        goal=f"""Provide the best recommendation for {group_data["destination"]}""",
        backstory=f"""You area  seasoned travel expert with deep knowledge of
        {group_data["destination"]}. You know the best spots, hidden gems, restaurants,
        and how to optimize time to see multiple attractions.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm
    )

    itinerary_planner = Agent(
        role="Itinerary Coordinator",
        goal="Create a balanced, day by day feasible itinerary with specific times",
        backstory = """You are a master plann who creates realisitic, well-paced itineraries with specific times.
        You consider travel time, opening time, rush hour, event schedules, and group energy levels to build
        perfect schedule.""",
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

    # --- TASKS ---
    event_search_task = Task(
        description=f"""
        1. Search for events and activities happening in {group_data['destination']} ({start_formatted} to {end_formatted}).
        2. CRITICAL: Search for venues/facilities for these PRIORITY interests: {priority_interests_str}.
        - PICKLEBALL (if present): Search specifically for:
            * "Best indoor pickleball courts in {group_data['destination']}"
            * "Pickleball open play {group_data['destination']}"
            * "Recreation center pickleball courts {group_data['destination']}"
            * "Indoor courts ratings/pickelball {group_data['destination']}"
            For each venue, capture: indoor/outdoor, court surface, reservation rules, open-play times, cost, and whether lights are available.
        - CUISINES (e.g., 'Asian Food'): Search specifically for:
            * "Best Asian restaurants in {group_data['destination']}"
            * "Best Chinese/Japanese/Korean restaurants in {group_data['destination']}"
            For each restaurant, capture: name, address, cuisine subtype, rating, price level, hours, and whether reservations are recommended.
        3. Prioritize free recreational activities first, but include paid options if they are highly rated or fit the group's budget.
        4. Return a list of candidate venues/events with links/sources for verification.
        """,
        agent=event_researcher,
        expected_output="A list of events AND list of specific venues (courts, parks, restaurants) that match the priority interests."
    )


    research_task = Task(
        description=f"""
        Using the events/venues found, select the BEST specific locations for the group.

        MANDATORY REQUIREMENT: You MUST provide a specific venue recommendation for EVERY item in this priority list: {priority_interests_str}.

        For each recommendation, include:
        - Name & Address
        - Operating Hours (verify they are open during {start_formatted})
        - Cost/Entry rules
        - Activities that complement the events found and match group interests
        - Practical logistics (locations, timing, costs)
        - Current weather consideration for those dates

        FALLBACK: If the Event Researcher did NOT return any valid options for a required interest (e.g., no 'Asian' restaurants or no 'indoor pickleball' venues),
        you MUST perform an additional targeted search (use different query formulations, include 'recreation center', 'indoor', 'open play', 'best rated') and supply at least one strong candidate.
        """,
        agent=local_expert,
        expected_output="A list of recommended specific places covering ALL priority interests.",
        context=[event_search_task]
    )


    planning_task = Task(
        description=f"""
        Create a detailed itinerary for {duration} days.

        STRICT RULES:
        1. Time Window: Schedule activities ONLY between {start_time} and {end_time}.
        2. MANDATORY INCLUSION: You MUST schedule an activity for EACH of these Priority Interests: {priority_interests_str}.
        (e.g., If 'Pickleball' is listed, you MUST schedule 'Pickleball at [Venue Name]' in the itinerary).
        3. Logic: Don't schedule outdoor sports at night unless the venue has lights.
        4. NO GENERIC PLACEHOLDERS: You are STRONGLY DISCOURAGED from writing "Dinner at a local restaurant" or "Lunch at a nearby cafe".
        HOWEVER, if the Local Travel Expert failed to provide a specific venue for a required cuisine or activity, you MAY perform a targeted search and select a named venue (include source links).
        5. Prioritize Free recreational activities first.
        Format as Markdown:
        # Itinerary for {group_data['destination']}
        ## DAY 1 - [Date]
        **[Start Time] - [End Time]: Activity Name**
        - Description...
        """,
        agent=itinerary_planner,
        expected_output="A complete day-by-day itinerary in Markdown format that includes ALL priority interests.",
        context=[event_search_task, research_task]
    )

    # --- CREW ---
    crew = Crew(
        agents=[event_researcher, local_expert, itinerary_planner],
        tasks=[event_search_task, research_task, planning_task],
        process=Process.sequential,
        max_rpm=10
    )
    result = crew.kickoff()
    # Convert result to string and strip code blocks if the LLM added them
    output_str = str(result)

    # Remove ```markdown and ``` fences
    cleaned_output = re.sub(r'```markdown', '', output_str, flags=re.IGNORECASE)
    cleaned_output = re.sub(r'```', '', cleaned_output)

    return cleaned_output.strip()
