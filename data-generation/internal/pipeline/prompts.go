package pipeline

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"time"

	"github.com/gouwsxander/slop-translator/data-generation/internal/models"
)

func BuildSummaryPrompt(datum models.Datum) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString("Extract the key points from the following paragraph.")
	promptBuilder.WriteString(
		fmt.Sprintf(" It is from the Wikipedia article \"%s\".", datum.PageTitle),
	)
	promptBuilder.WriteString(" Focus on the main facts, concepts, and relationships.")
	promptBuilder.WriteString(" Present your summary as a concise list of key points.")
	promptBuilder.WriteString("\n\nWikipedia paragraph:\n")
	promptBuilder.WriteString(datum.Human)
	promptBuilder.WriteString("\n\nGive just a bullet list of the key points — do not include other text.")

	return promptBuilder.String()
}

func BuildRewritePrompt(datum models.Datum) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString(
		fmt.Sprintf("You are writing a paragraph for an article on \"%s\"", datum.PageTitle),
	)
	promptBuilder.WriteString(" Using the key points provided below, write a cohesive paragraph in Wikipedia's encyclopedic style.")
	promptBuilder.WriteString("\n\nRequirements:")
	promptBuilder.WriteString("\n- Use formal, neutral, encyclopedic tone")
	promptBuilder.WriteString("\n- Write in third person")
	promptBuilder.WriteString("\n- Present information objectively")
	promptBuilder.WriteString("\n- Create smooth transitions between ideas")
	promptBuilder.WriteString("\n- Do NOT copy phrases verbatim from the key points — rephrase naturally")
	promptBuilder.WriteString("\n- Do NOT include any markdown formatting (e.g., do NOT use *italics* and do NOT use **bold**.")
	promptBuilder.WriteString("\n\nKey points to cover:\n")
	promptBuilder.WriteString(datum.Summary)
	promptBuilder.WriteString("\n\nGive just your written paragraph — do not include other text in your response.")

	return promptBuilder.String()
}

func MakeRequests(preprompts <-chan models.Datum, builder func(models.Datum) string, languageModels []string) <-chan models.Datum {
	requests := make(chan models.Datum)

	go func() {
		defer close(requests)

		for datum := range preprompts {
			languageModel := randomChoice(languageModels)
			prompt := builder(datum)

			datum.Request = &models.Request{
				Model:   languageModel,
				Prompt:  prompt,
				Retries: 0,
				NextTry: time.Now(),
			}

			requests <- datum
		}
	}()

	return requests
}

func randomChoice[T any](s []T) T {
	n := len(s)
	idx := rand.IntN(n)
	return s[idx]
}
