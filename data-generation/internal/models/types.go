package models

import "time"

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
