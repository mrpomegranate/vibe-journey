from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from typing import List, Dict

from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()
search_tool = SerperDevTool()
llm = LLM(model="gemini/gemini-2.0-flash",
          temperature=0)

# Sample collective group interest
group_data = {
    "people": [
        {"name": "Samantha", "interests": ["museums", "art","food"]},
        {"name": "John", "interests": ["food", "nightlife","comedy"]},
        {"name": "Kyle", "interests": ["pickleball", "nightlife", "asian food"]},
    ],
    "destination": "Northern Virginia",
    "start_date": "2025-10-06",
    "end_date": "2025-10-06",
    "budget": "moderate"
}

def calculate_trip_duration(start_date: str, end_date: str) -> int:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    duration = (end - start).days + 1

    dates = []
    for i in range(duration):
        day = start + timedelta(days=i)
        dates.append(day.strftime("%Y-%m-%d"))

    return duration, dates, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

def aggregated_interests(people: List[Dict]) -> List[str]:

    all_interests = []
    interest_count = {}

    for person in people:
        for interest in person["interests"]:
            all_interests.append(interest)
            interest_count[interest]= interest_count.get(interest, 0) + 1

    common_interests = sorted(interest_count.items(), key=lambda x: x[1], reverse=True)
    unique_interests = list(set(all_interests))

    interest ={
        "all_interests": unique_interests,
        "common_interests": [interest for interest, count in common_interests if count > 1],
        "unique_interests": [interest for interest, count in common_interests if count == 1],
        "interest_summary": f"Group interests include: {', '.join(unique_interests)}"
    }

    return interest


def create_itinerary_crew(group_data: Dict) -> Crew:

    duration, date_list, start_formatted, end_formatted = calculate_trip_duration(group_data["start_date"],
                                                                                  group_data["end_date"])

    search_tool = SerperDevTool()
    aggregate_interests = aggregated_interests(group_data['people'])

    event_researcher = Agent(
        role = "Real-Time Event and Activity Researcher",
        goal = f"""Find free activity, concerts, museum exhibition, shows, events in
        {group_data['destination']} during the specific dates based on group interest""",
        backstory = f"""You are an expert at finding current events, concerts, museum
        exhibitions, theater shows, and special happenings that match group interests.
        You search for real-time information about what's actually happening during the travel dates.""",
        verbose = True,
        allow_delegation = False,
        tools = [search_tool],
        llm=llm
    )

    local_expert = Agent(
        role = "Local Travel Expert",
        goal = f"""Provide the best recommendation for {group_data["destination"]}""",
        backstory = f"""You area  seasoned travel expert with deep knowledge of
        {group_data["destination"]}. You know the best spots, hidden gems, restaurants,
        and how to optimize time to see multiple attractions.""",
        verbose = True,
        allow_delegation = False,
        tools = [search_tool],
        llm=llm
    )

    itinerary_planner = Agent(
        role = "Itinerary Coordinator",
        goal="Create a balanced, day by day feasible itinerary with specific times",
        backstory = """You are a master plann who creates realisitic, well-paced itineraries with specific times.
        You consider travel time, opening time, rush hour, event schedules, and group energy levels to build
        perfect schedule""",
        verbose = True,
        allow_delegation = False,
        llm=llm
    )


    event_search_task = Task(
        description=f"""Search for actual activity, events, concerts, shows, and exhibitions
        happening in {group_data['destination']} during these specific dates:
        {', '. join(date_list)} ({start_formatted} to {end_formatted})

        Group interests: {aggregate_interests['interest_summary']}
        Common interests: {', '.join(aggregate_interests['common_interests'])}
        Unique interests: {', '.join(aggregate_interests['unique_interests'])}

        Search for events matching these interests:
        {chr(10).join([f"- {interest.capitalize()} related events" for interest in aggregate_interests['all_interests']])}


        Prioritize events that match common interests (enjoyed by multiple people).

        For each event/activity found include:
        - Event name and type
        - Exact date and time
        - Venue and location
        - Ticket information if available
        """,
        agent=event_researcher,
        expected_output="A comprehensive list of real events happening during the travel dates, organized by interest category with full details"
    )

    research_task = Task(
        description=f"""Based on the events found and group interests, research and recommend specific attractions, restaurants, public park, and activities in
        {group_data['destination']} with a {group_data['budget']}.

        Group Interests: {aggregate_interests['interest_summary']}

        Focus on:
        - Operating hours for museums and attractions during {start_formatted}
        - Restaurants suitable for the group (check current availability)
        - Activities that complement the events found and match group interests
        - Practical logistics (locations, timing, costs)
        - Current weather consideration for those dates
        """,
        agent=local_expert,
        expected_output="A comprehensive list of recommended places with hours, locations, and practical details",
        context=[event_search_task]
    )

    planning_task = Task(
        description=f"""Create a detailed day-by-day itinerary for {duration}
        from {start_formatted} to {end_formatted}.

        GROUP INTERESTS TO BALANCE: {aggregate_interests['interest_summary']}
        - Prioritize common interests: {', '.join(aggregate_interests['common_interests']) if aggregate_interests['common_interests'] else 'None'}
        - Include unique interests: {', '.join(aggregate_interests['unique_interests']) if aggregate_interests['unique_interests'] else 'None'}

        For each date({', '.join(date_list)}), createa schedule that:
        1. Includes SPECIFIC Times (e.g., "9:00 AM - 11:00PM")
        2. Incorporate the real events found (concerts, shows, exhibition, park to play pickleball and other sports)
        3. Balances different interests throughout each day
        4. Groups nearby activities together logically
        5. Includes addresses and venue names
        6. Plans around event times (if concert is at 8 PM, plan accordingly. Take traffic into account)
        7. Includes meal times with restaurant suggestions
        8. Allows travel time between locations
        9. Provides alternatives for flexibility

        Format as:
        DAY 1 - [Date]
        Morning (9:00 AM - 12:00 PM):
        - Activity with time, location, description

        Afternoon (12:00 PM - 5:00 PM):
        - Activities with times

        Evening (5:00 PM - 10:00 PM):
        - Activities/events with times

        Repeat for each day.
        """,
        agent=itinerary_planner,
        expected_output="A complete day-by-day itinerary with specific times, dates, and locations",
        context=[event_search_task, research_task]
    )


    crew = Crew(
        agents=[event_researcher, local_expert, itinerary_planner],
        tasks=[event_search_task, research_task, planning_task],
        process=Process.sequential
    )

    return crew


def main():
    """Run the itinerary builder"""
    duration, date_list, start_formatted, end_formatted = calculate_trip_duration(
        group_data['start_date'],
        group_data['end_date']
    )

    print("🗺️  Starting Group Itinerary Builder with Real-Time Event Search...")
    print(f"\nGroup: {len(group_data['people'])} people")
    print(f"Destination: {group_data['destination']}")
    print(f"Dates: {start_formatted} to {end_formatted} ({duration} days)")
    print(f"Specific dates: {', '.join(date_list)}")
    print(f"Budget: {group_data['budget']}")
    print("\nGroup Interests:")
    for person in group_data['people']:
        print(f"  - {person['name']}: {', '.join(person['interests'])}")

    print("\n" + "="*70)
    print("Building your itinerary with real-time event search...\n")

    # Create and run crew
    crew = create_itinerary_crew(group_data)
    result = crew.kickoff()

    print("\n" + "="*70)
    print("✅ FINAL ITINERARY WITH EVENTS")
    print("="*70)
    print(result)

    return result


if __name__ == "__main__":
    # To customize, modify the group_data dictionary above
    # Change start_date and end_date to your desired dates (YYYY-MM-DD format)
    main()