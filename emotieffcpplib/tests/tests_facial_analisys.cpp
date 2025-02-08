#include <gtest/gtest.h>
#include <emotiefflib/facial_analysis.h>

TEST(EmotionRecognition, Basic) {
    EmotiEffLibRecognizerTorch("asdfsf", "sdfsdf");
  // Expect two strings not to be equal.
  EXPECT_STRNE("hello", "world");
  // Expect equality.
  EXPECT_EQ(7 * 6, 42);
}
