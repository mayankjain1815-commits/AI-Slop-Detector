package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

type ScrapedDatum struct {
	PageTitle string `json:"page_title"`
	Text      string `json:"text"`
}

type Datum struct {
	Idx          int
	PageTitle    string
	Human        string
	AI           string
	Summary      string
	LastRequest  *Request
	LastResponse *Response
}

type Request struct {
	Model   string
	Prompt  string
	Retries int
	LastTry time.Time
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
}

func NewPipeline(inputFilePath, outputFilePath string, requestConcurrency int) *Pipeline {
	ctx, cancel := context.WithCancel(context.Background())
	return &Pipeline{
		inputFilePath:      inputFilePath,
		outputFilePath:     outputFilePath,
		preprompts:         make(chan Datum, 2),
		requests:           make(chan Datum, 2),
		responses:          make(chan Datum, 2),
		data:               make(chan Datum, 2),
		ctx:                ctx,
		cancel:             cancel,
		maxRetries:         3,
		requestConcurrency: requestConcurrency,
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

func main() {
	pipeline := NewPipeline("../scrapes/scrape_1763655781666449000.jsonl", ".jsonl", 2)

	go pipeline.readInputFile()
	go func() {
		for datum := range pipeline.preprompts {
			fmt.Printf("%d %s\n", datum.Idx, datum.PageTitle)
		}
	}()

	time.Sleep(50 * time.Millisecond)
}
