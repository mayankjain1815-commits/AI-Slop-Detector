package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"strings"
	"sync"
	"time"
)

type ScrapedDatum struct {
	PageTitle string `json:"page_title"`
	Text      string `json:"text"`
}

type Datum struct {
	Idx       int
	PageTitle string
	Human     string
	AI        string
	Summary   string
	Request   *Request
	Response  *Response
}

type Request struct {
	Model   string
	Prompt  string
	Retries int
	NextTry time.Time
}

type Response struct {
	Success bool
	Content string // Place holder for now. Later this should model OpenRouter response?
	Error   error
}

type Pipeline struct {
	inputFilePath      string
	outputFilePath     string
	preprompts         chan Datum
	requests           chan Datum
	responses          chan Datum
	data               chan Datum
	wg                 sync.WaitGroup
	ctx                context.Context
	cancel             context.CancelFunc
	maxRetries         int
	requestConcurrency int
	models             []string
}

func NewPipeline(inputFilePath, outputFilePath string, requestConcurrency int) *Pipeline {
	ctx, cancel := context.WithCancel(context.Background())
	return &Pipeline{
		inputFilePath:      inputFilePath,
		outputFilePath:     outputFilePath,
		preprompts:         make(chan Datum, 8),
		requests:           make(chan Datum, 8),
		responses:          make(chan Datum, 8),
		data:               make(chan Datum, 8),
		ctx:                ctx,
		cancel:             cancel,
		maxRetries:         3,
		requestConcurrency: requestConcurrency,
		models:             []string{"Model A", "Model B"},
	}
}

func (p *Pipeline) Run() error {
	return nil
}

func (p *Pipeline) readInputFile() error {
	file, err := os.Open(p.inputFilePath)
	if err != nil {
		return fmt.Errorf("Unable to open input file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	idx := 0

	for scanner.Scan() {
		idx++
		var scrapedDatum ScrapedDatum
		err := json.Unmarshal(scanner.Bytes(), &scrapedDatum)
		if err != nil {
			log.Printf("Failed to parse line %d: %v", idx, err)
			continue
		}

		datum := Datum{
			Idx:       idx,
			PageTitle: scrapedDatum.PageTitle,
			Human:     scrapedDatum.Text,
		}

		select {
		case <-p.ctx.Done():
			return nil
		case p.preprompts <- datum:
		}
	}

	return scanner.Err() // nil if reaches EOF
}

func (p *Pipeline) makeRequests() {
	for datum := range p.preprompts {
		model := randomChoice(p.models)

		var promptBuilder strings.Builder
		if datum.Summary == "" {
			// Generate summary prompt
			promptBuilder.WriteString("Extract the key points from the following paragraph.")
			promptBuilder.WriteString(
				fmt.Sprintf(" It is from the Wikipedia article \"%s\".", datum.PageTitle),
			)
			promptBuilder.WriteString(" Focus on the main facts, concepts, and relationships.")
			promptBuilder.WriteString(" Present your summary as a concise list of key points.")
			promptBuilder.WriteString("\n\nWikipedia paragraph:\n")
			promptBuilder.WriteString(datum.Human)
			promptBuilder.WriteString("\n\nGive just a bullet list of the key points — do not include other text.")
		} else if datum.AI == "" {
			// Generate rewrite prompt
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
		} else {
			// Unexpected!!
			log.Printf("Unexpected! Datum index %d with summary and AI text was passed through preprompt...\n", datum.Idx)
			p.data <- datum
		}

		datum.Request = &Request{
			Model:   model,
			Prompt:  promptBuilder.String(),
			Retries: 0,
			NextTry: time.Now(),
		}

		p.requests <- datum
	}
}

func randomChoice[T any](s []T) T {
	n := len(s)
	idx := rand.IntN(n)
	return s[idx]
}

func main() {
	pipeline := NewPipeline("../scrapes/scrape_1763655781666449000.jsonl", ".jsonl", 2)

	go pipeline.readInputFile()
	go pipeline.makeRequests()

	go func() {
		for datum := range pipeline.requests {
			fmt.Println(datum.Request.Prompt)
			fmt.Println("========")

			if datum.Summary == "" {
				datum.Summary = "<summary here>"
				go func() {
					pipeline.preprompts <- datum
				}()
			}
		}
	}()

	time.Sleep(20 * time.Second)
}
