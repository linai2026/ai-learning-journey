class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # Create a dictionary to hold the grouped anagrams
        anagrams = {}
        
        for word in strs:
            # Sort the word to create a key
            sorted_word = ''.join(sorted(word)) 
            
            if sorted_word not in anagrams:
                # Initialize the list for this sorted word
                anagrams[sorted_word] = []

            # Append the original word to the corresponding list
            anagrams[sorted_word].append(word)
        
        return list(anagrams.values())
        