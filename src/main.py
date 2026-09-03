import langchain_community as lc


def main():
    # Create a new instance of the LangChain community
    community = lc.Community()

    # Add a new member to the community
    member = lc.Member(name="John Doe", role="Developer")
    community.add_member(member)

    # List all members in the community
    members = community.list_members()
    for m in members:
        print(f"Member Name: {m.name}, Role: {m.role}")