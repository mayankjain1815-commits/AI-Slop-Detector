package pipeline

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/gouwsxander/slop-translator/data-generation/internal/models"
)

type scrapedDatum struct {
	PageTitle string `json:"page_title"`
	Text      string `json:"text"`
}

func StreamInputFile(inputFilePath string, ctx context.Context) <-chan models.Datum {
	preprompts := make(chan models.Datum)

	go func() {
		defer close(preprompts)
		file, err := os.Open(inputFilePath)
		if err != nil {
			fmt.Printf("Unable to open input file: %v", err)
			return
		}
		defer file.Close()

		scanner := bufio.NewScanner(file)
		idx := 0

		for scanner.Scan() {
			idx++
			var scrapedDatum scrapedDatum
			err := json.Unmarshal(scanner.Bytes(), &scrapedDatum)
			if err != nil {
				log.Printf("Failed to parse line %d: %v", idx, err)
				continue
			}

			datum := models.Datum{
				Idx:       idx,
				PageTitle: scrapedDatum.PageTitle,
				Human:     scrapedDatum.Text,
			}

			select {
			case <-ctx.Done():
				return
			case preprompts <- datum:
			}
		}

		if scanner.Err() != nil {
			log.Printf("Failed to scan file %d: %v", idx, scanner.Err())
			return
		}
	}()

	return preprompts
}
