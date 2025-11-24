package main

import (
	"context"
	"fmt"

	"github.com/gouwsxander/slop-translator/data-generation/internal/pipeline"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())

	preprompts := pipeline.StreamInputFile("../scrapes/scrape_1763655781666449000.jsonl", ctx)
	summaryRequests := pipeline.MakeRequests(preprompts, pipeline.BuildSummaryPrompt, []string{"Model A", "Model B"})

	c := 0
	for datum := range summaryRequests {
		c++
		fmt.Println(datum.Request.Prompt)
		fmt.Println("========")

		if c > 5 {
			cancel()
		}
	}
}
