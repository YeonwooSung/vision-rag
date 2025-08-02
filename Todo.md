# ToDo Lists

- [ ] Add support for AWS S3 for image upload
- [x] Add support for MUVERA for faster search
    - ColPali model is a multi-vector model, which generates embeddings per patches.
    - Vector search over all patch-embeddings is definitely a bad idea.
    - Use MUVERA to quantize the multi-vectors into fixed-sized single vector!
