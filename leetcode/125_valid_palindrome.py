class Solution:

    # One-liner approach
    def isPalindrome(self, s: str) -> bool:
        # Convert the string to lowercase and filter out non-alphanumeric characters
        filtered_chars = [char.lower() for char in s if char.isalnum()]
        
        # Check if the filtered list of characters is equal to its reverse
        return filtered_chars == filtered_chars[::-1]

    # Two-pointer approach
    def isPalindromeTwoPointers(self, s: str) -> bool:
        # Initialize two pointers
        left, right = 0, len(s) - 1
        
        while left < right:
            # Move the left pointer to the next alphanumeric character
            while left < right and not s[left].isalnum():
                left += 1
            # Move the right pointer to the previous alphanumeric character
            while left < right and not s[right].isalnum():
                right -= 1
            
            # Compare the characters at the left and right pointers
            if s[left].lower() != s[right].lower():
                return False
            
            # Move both pointers towards the center
            left += 1
            right -= 1
        
        return True