# StableFast3D.Net

**Powered by Stability AI.** C# port (.NET 8 + TorchSharp) of Stable Fast 3D, inference only: one picture in, one binary glTF out. It will replace the Python container of Veritmo as the product's engine for these models. **Status: ported** (the service makes models; see the README for what is still open).

## Articles

- [Porting plan](porting-plan.md): the model stage by stage, what each stage is in the reference and where it goes here, and how it is checked.
- [The HTTP service](http-service.md): the contract, the same as the Python container's.

## API reference

The public API of the `StableFast3D` library, generated from the XML comments of the source.
