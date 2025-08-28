"""
These functions are used in the post-response generation step for traceability of sources and structured in-text citation management

Algother, these steps help to mitigate source hallucination in IPOSGPT
"""

# Extract referenced source numbers from generated response
def extract_referenced_sources(response_text):
    """Extract all source numbers referenced in the response text."""
    import re
    # Find all source references in the format [X] or [X, Y, Z]
    references = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', response_text)
    
    # Process each reference and extract individual source numbers
    referenced_sources = set()
    for ref in references:
        # Split by comma if there are multiple sources in one citation
        for source_num in re.split(r',\s*', ref):
            try:
                referenced_sources.add(int(source_num))
            except ValueError:
                pass  # Skip if not a valid integer
    
    return referenced_sources



# Renumber sources and update that in the generated response
def renumber_sources_and_update_response(response_text, referenced_source_numbers, source_number_to_source):
    # Create a mapping from old source numbers to new sequential ones
    old_to_new_mapping = {}
    new_source_num = 1
    
    for old_source_num in sorted(referenced_source_numbers):
        old_to_new_mapping[old_source_num] = new_source_num
        new_source_num += 1
    
    # Create a new mapping for source display
    new_source_number_to_source = {}
    for old_num, new_num in old_to_new_mapping.items():
        if old_num in source_number_to_source:
            new_source_number_to_source[new_num] = source_number_to_source[old_num]
    
    # Update references in the response text using regex
    import re
    updated_response = response_text
    
    # Handle multi-source citations like [1, 3, 5]
    def replace_citation(match):
        citation_content = match.group(1)
        citation_numbers = [int(num.strip()) for num in citation_content.split(',')]
        new_citation_numbers = [old_to_new_mapping.get(num, num) for num in citation_numbers]
        return f"[{', '.join(str(num) for num in new_citation_numbers)}]"
    
    # Replace citations in the text
    updated_response = re.sub(r'\[([0-9\s,]+)\]', replace_citation, updated_response)
    
    return updated_response, new_source_number_to_source


####################################################################################################
#LLM Initial response
response_text = "LLM initial response"

#Source list and URL to source number mapping
sources = ["List of unique sources in the retrieval process"]
url_to_source_number = {} # url to source_number mapping

# Extract the source numbers that were actually referenced in the response
referenced_source_numbers = extract_referenced_sources(response_text)

# Create a mapping from source number to source object
source_number_to_source = {}
for source in sources:
    url = source["url"]
    if url in url_to_source_number:
        source_num = url_to_source_number[url]
        source_number_to_source[source_num] = source
    
# Source renumbering in the response
updated_response, new_source_number_to_source = renumber_sources_and_update_response(
    response_text, 
    referenced_source_numbers, 
    source_number_to_source)
            

if new_source_number_to_source:
    for source_num in sorted(new_source_number_to_source.keys()):
        source = new_source_number_to_source[source_num]
        
        # Add the source number to the display
        print(f"### Source {source_num}")
        
        # Clickable title with URL
        print(f"#### [{source['title']}]({source['url']})")
        
        # Rest of the source information display
        if source['authors']:
            print(f"**Authors:** {source['authors']}")
        
        print(f"**Date:** {source['date']}")
        print(f"**Knowledge Category:** {source['knowledge_category']}")
        print(f"**Publisher:** {source['publisher']}")
        print(f"**Data Source:** {source['data_source']}")
        print(f"**Source Type:** {source['source_type']}")
        
        peer_review_status = "Peer Reviewed" if source['is_peer_reviewed'] else "Not Peer Reviewed"
        print(f"**Peer Review Status:** {peer_review_status}")
        
        print("---")  # Separator between sources
else:
    print("No sources were referenced in the response.")