import json
import re
from llm_helper import llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException


def clean_unicode_text(text):
    """Clean problematic Unicode characters that cause encoding errors"""
    if not isinstance(text, str):
        return text
    
    try:
        # Remove surrogate characters that cause UTF-8 encoding issues
        text = re.sub(r'[\ud800-\udfff]', '', text)
        
        # Handle common escape sequences
        text = text.replace('\\n', '\n').replace('\\t', '\t')
        
        # Ensure UTF-8 compatibility
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text
    except Exception as e:
        print(f"Unicode cleaning error: {e}")
        # Fallback: remove all problematic characters
        return re.sub(r'[^\x00-\x7F\u0080-\uFFFF]', ' ', str(text))


def process_posts(raw_file_path, processed_file_path=None):
    """Process LinkedIn posts and extract metadata"""
    try:
        with open(raw_file_path, encoding='utf-8') as file:
            posts = json.load(file)
            enriched_posts = []
            
            for i, post in enumerate(posts):
                print(f"Processing post {i+1}/{len(posts)}...")
                
                # Clean the text before processing
                cleaned_text = clean_unicode_text(post['text'])
                post['text'] = cleaned_text
                
                try:
                    metadata = extract_metadata(cleaned_text)
                    post_with_metadata = post | metadata
                    enriched_posts.append(post_with_metadata)
                except Exception as e:
                    print(f"Error processing post {i+1}: {e}")
                    # Add post without metadata if extraction fails
                    post['line_count'] = len(cleaned_text.split('\n'))
                    post['language'] = 'English'  # Default
                    post['tags'] = ['General']    # Default
                    enriched_posts.append(post)

        print("Unifying tags...")
        unified_tags = get_unified_tags(enriched_posts)
        
        # Apply unified tags
        for post in enriched_posts:
            current_tags = post.get('tags', [])
            new_tags = {unified_tags.get(tag, tag) for tag in current_tags}
            post['tags'] = list(new_tags)

        # Save processed data
        if processed_file_path:
            with open(processed_file_path, encoding='utf-8', mode="w") as outfile:
                json.dump(enriched_posts, outfile, indent=4, ensure_ascii=False)
            print(f"Processed {len(enriched_posts)} posts saved to {processed_file_path}")
        
        return enriched_posts
        
    except FileNotFoundError:
        print(f"Error: File {raw_file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {raw_file_path}: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def extract_metadata(post):
    """Extract metadata from a LinkedIn post using LLM"""
    # Clean the post text before sending to LLM
    cleaned_post = clean_unicode_text(post)
    
    template = '''
    You are given a LinkedIn post. You need to extract number of lines, language of the post and tags.
    1. Return a valid JSON. No preamble. 
    2. JSON object should have exactly three keys: line_count, language and tags. 
    3. tags is an array of text tags. Extract maximum two tags.
    4. Language should be English or Hinglish (Hinglish means hindi + english)
    
    Here is the actual post on which you need to perform this task:  
    {post}
    '''

    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    
    try:
        response = chain.invoke(input={"post": cleaned_post})
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
        
        # Validate the response structure
        if not all(key in res for key in ['line_count', 'language', 'tags']):
            raise ValueError("Missing required keys in LLM response")
            
        # Ensure tags is a list
        if not isinstance(res['tags'], list):
            res['tags'] = [res['tags']] if res['tags'] else ['General']
            
        return res
        
    except OutputParserException as e:
        print(f"JSON parsing error: {e}")
        raise OutputParserException("Context too big. Unable to parse jobs.")
    except Exception as e:
        print(f"LLM processing error: {e}")
        # Return default metadata
        return {
            "line_count": len(cleaned_post.split('\n')),
            "language": "English",
            "tags": ["General"]
        }


def get_unified_tags(posts_with_metadata):
    """Unify and merge similar tags using LLM"""
    unique_tags = set()
    
    # Loop through each post and extract the tags
    for post in posts_with_metadata:
        if 'tags' in post and isinstance(post['tags'], list):
            unique_tags.update(post['tags'])

    if not unique_tags:
        return {}
        
    unique_tags_list = ','.join(unique_tags)
    print(f"Found {len(unique_tags)} unique tags: {unique_tags_list}")

    template = '''I will give you a list of tags. You need to unify tags with the following requirements,
    1. Tags are unified and merged to create a shorter list. 
       Example 1: "Jobseekers", "Job Hunting" can be all merged into a single tag "Job Search". 
       Example 2: "Motivation", "Inspiration", "Drive" can be mapped to "Motivation"
       Example 3: "Personal Growth", "Personal Development", "Self Improvement" can be mapped to "Self Improvement"
       Example 4: "Scam Alert", "Job Scam" etc. can be mapped to "Scams"
    2. Each tag should follow title case convention. example: "Motivation", "Job Search"
    3. Output should be a JSON object, No preamble
    4. Output should have mapping of original tag and the unified tag. 
       For example: {{"Jobseekers": "Job Search",  "Job Hunting": "Job Search", "Motivation": "Motivation"}}
    
    Here is the list of tags: 
    {tags}
    '''
    
    pt = PromptTemplate.from_template(template)
    chain = pt | llm
    
    try:
        response = chain.invoke(input={"tags": unique_tags_list})
        json_parser = JsonOutputParser()
        res = json_parser.parse(response.content)
        
        print(f"Unified {len(unique_tags)} tags into {len(set(res.values()))} categories")
        return res
        
    except OutputParserException as e:
        print(f"Tag unification parsing error: {e}")
        # Return identity mapping as fallback
        return {tag: tag for tag in unique_tags}
    except Exception as e:
        print(f"Tag unification error: {e}")
        return {tag: tag for tag in unique_tags}


if __name__ == "__main__":
    print("Starting LinkedIn post processing...")
    processed_posts = process_posts("data/raw_posts.json", "data/processed_posts.json")
    print(f"Processing complete! Processed {len(processed_posts)} posts.")
    
    # Print sample results
    if processed_posts:
        print("\nSample processed post:")
        sample_post = processed_posts[0]
        print(f"Text preview: {sample_post['text'][:100]}...")
        print(f"Line count: {sample_post.get('line_count', 'N/A')}")
        print(f"Language: {sample_post.get('language', 'N/A')}")
        print(f"Tags: {sample_post.get('tags', 'N/A')}")
        print(f"Engagement: {sample_post.get('engagement', 'N/A')}")