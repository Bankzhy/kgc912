  public int lengthOfLongestSubstring(String s) {
    int start = 0;
    int maxLen = 0;
    HashMap<Character, Integer> map = new HashMap<>();

    // Iterate through the string
    for (int end = 0; end < s.length(); end++) {
        char ch = s.charAt(end);
        if (map.containsKey(ch) && map.get(ch) >= start) {
           start = map.get(ch) + 1;
        }
        maxLen = Math.max(maxLen, end - start + 1);
        map.put(ch, end);
      }

      return maxLen;
  }
