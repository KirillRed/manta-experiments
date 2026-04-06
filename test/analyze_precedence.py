import os
from collections import defaultdict

def find_action_dependencies(folder_path):
    # target_action_counts[X] = total number of videos where action X appears
    action_video_counts = defaultdict(int)
    
    # precedes_counts[X][Y] = number of videos where action Y appears before action X
    precedes_counts = defaultdict(lambda: defaultdict(int))
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return

    # Process all .txt files
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    if not txt_files:
        print(f"❌ No .txt files found in {folder_path}")
        return
        
    print(f"📂 Analyzing {len(txt_files)} files...\n")

    for filename in txt_files:
        filepath = os.path.join(folder_path, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if not lines:
            continue
            
        # 1. Collapse consecutive identical frames
        collapsed_seq = [lines[0]]
        for action in lines[1:]:
            if action != collapsed_seq[-1]:
                collapsed_seq.append(action)
                
        # 2. Count how many videos this action appears in (Denominator)
        unique_actions_in_video = set(collapsed_seq)
        for action in unique_actions_in_video:
            action_video_counts[action] += 1
            
        # 3. Find all (Y precedes X) pairs in this video
        seen_actions = set()
        pairs_found_in_video = set()
        
        for current_action in collapsed_seq:
            for past_action in seen_actions:
                # We only care about different actions
                if past_action != current_action:
                    # Save the pair. Using a set prevents double-counting 
                    # if the sequence is A -> B -> A -> B
                    pairs_found_in_video.add((past_action, current_action))
            
            # Add current action to seen so it can be a "past_action" for future frames
            seen_actions.add(current_action)
            
        # 4. Record the pairs for this video (Numerator)
        for past_action, target_action in pairs_found_in_video:
            precedes_counts[target_action][past_action] += 1

    # Print the final statistics
    print("====== ACTION DEPENDENCY RESULTS ======\n")
    
    # Sort target actions alphabetically for easier reading
    for target_action in sorted(action_video_counts.keys()):
        total_occurrences = action_video_counts[target_action]
        print(f"🎯 Action: '{target_action}' (Found in {total_occurrences} videos)")
        
        # Calculate percentages
        preceding_stats = []
        for preceding_action, pre_count in precedes_counts[target_action].items():
            percentage = (pre_count / total_occurrences) * 100
            preceding_stats.append((preceding_action, percentage))
            
        # Sort by highest percentage first
        preceding_stats.sort(key=lambda item: item[1], reverse=True)
        
        if not preceding_stats:
            print("  -> (Usually the very first action, preceded by nothing)")
        else:
            for pre_action, pct in preceding_stats:
                # Optional: You can add an 'if pct > 50:' here if you only want strong dependencies
                print(f"  <- {pct:6.2f}% preceded by: '{pre_action}'")
                
        print("-" * 50)

# --- EXECUTE ---
if __name__ == '__main__':
    # REPLACE THIS with the path to your folder containing the .txt files
    FOLDER_PATH = "/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/groundTruth" 
    
    find_action_dependencies(FOLDER_PATH)