# AI Squad

Notes on the AI Squad work for the Face recognition. The [DeepFace](https://github.com/serengil/deepface) framework is explored for this study. 

```mermaid
graph LR;
A[Inputs in S3 Bucket] -->C[EC2 instance GPU/CPU];
    B[Docker Image in AWS ECR] -->C;
    C -->D[Output in S3 bucket];
```
