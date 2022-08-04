package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"reflect"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"github.com/sugarme/tokenizer/pretrained"
)

func init() { log.SetFlags(log.Lshortfile | log.LstdFlags) }

func convertToF(ar []int) []byte {
	newar := make([]byte, len(ar))
	var v int
	var i int
	for i, v = range ar {
		newar[i] = byte(v)
	}
	return newar
}

func main() {

	tk := pretrained.BertBaseUncased()

	sentence := `The Gophers craft code using [MASK] language.`
	en, err := tk.EncodeSingle(sentence)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(reflect.TypeOf(en.Ids))
	fmt.Printf("ids: %q\n", en.Ids)
	fmt.Printf("typeids: %q\n", en.TypeIds)
	fmt.Printf("mask: %q\n", en.AttentionMask)

	// Output
	// tokens: ["the" "go" "##pher" "##s" "craft" "code" "using" "[MASK]" "language" "."]
	// offsets: [[0 3] [4 6] [6 10] [10 11] [12 17] [18 22] [23 28] [29 35] [36 44] [44 45]]
	// START SIMPLE
	// Create a backend receiver
	backend := gorgonnx.NewGraph()
	//backend := simple.NewSimpleGraph()
	// Create a model and set the execution backend
	model := onnx.NewModel(backend)

	// read the onnx model
	b, _ := ioutil.ReadFile("onnx-conversion/onnx/embeddings.onnx")
	// Decode it into the model
	err = model.UnmarshalBinary(b)
	if err != nil {
		log.Fatal(err)
	}
	//Set the first input, the number depends of the model
	inputA0, err := onnx.NewTensor(convertToF(en.Ids))
	inputA1, err := onnx.NewTensor(convertToF(en.TypeIds))
	inputA2, err := onnx.NewTensor(convertToF(en.AttentionMask))

	fmt.Println(inputA0.Data())
	model.SetInput(0, inputA0)
	model.SetInput(1, inputA1)
	model.SetInput(2, inputA2)
	err = backend.Run()
	if err != nil {
		fmt.Println("aaaaaaaaa")
		log.Fatal(err)
	}
	// Check error
	output, _ := model.GetOutputTensors()
	// write the first output to stdout
	fmt.Println(output[0])
}
