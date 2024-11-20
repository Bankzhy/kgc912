 protected synchronized void activateScavenger() {
        int initialSize = vocabulary.size();
        List<VocabularyWord> words = new ArrayList<>(vocabulary.values());
        for (VocabularyWord word : words) {
            // scavenging could be applied only to non-special tokens that are below minWordFrequency
            if (word.isSpecial() || word.getCount() >= minWordFrequency || word.getFrequencyShift() == null) {
                word.setFrequencyShift(null);
                continue;
            }

            // save current word counter to byte array at specified position
            word.getFrequencyShift()[word.getRetentionStep()] = (byte) word.getCount();

            /*
                    we suppose that we're hunting only low-freq words that already passed few activations
                    so, we assume word personal threshold as 20% of minWordFrequency, but not less then 1.

                    so, if after few scavenging cycles wordCount is still <= activation - just remove word.
                    otherwise nullify word.frequencyShift to avoid further checks
              */
            int activation = Math.max(minWordFrequency / 5, 2);
            logger.debug("Current state> Activation: [" + activation + "], retention info: "
                            + Arrays.toString(word.getFrequencyShift()));
            if (word.getCount() <= activation && word.getFrequencyShift()[this.retentionDelay - 1] > 0) {

                // if final word count at latest retention point is the same as at the beginning - just remove word
                if (word.getFrequencyShift()[this.retentionDelay - 1] <= activation
                                && word.getFrequencyShift()[this.retentionDelay - 1] == word.getFrequencyShift()[0]) {
                    vocabulary.remove(word.getWord());
                }
            }

            // shift retention history to the left
            if (word.getRetentionStep() < retentionDelay - 1) {
                word.incrementRetentionStep();
            } else {
                for (int x = 1; x < retentionDelay; x++) {
                    word.getFrequencyShift()[x - 1] = word.getFrequencyShift()[x];
                }
            }
        }
        logger.info("Scavenger was activated. Vocab size before: [" + initialSize + "],  after: [" + vocabulary.size()
                        + "]");
    }