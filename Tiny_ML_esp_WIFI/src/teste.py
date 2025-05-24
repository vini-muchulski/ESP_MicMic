#!/usr/bin/env python3
"""
Converte um arquivo .tflite em:
  • modelo_seno_int8.cc  (array const uint8_t)
  • modelo_seno_int8.h   (extern … e tamanho)

Uso:
  python gerar_firmware_array.py  \
         --input  modelo_seno_int8.tflite \
         --name   modelo_seno_int8        \
         --outdir .

Author: Apeiron
"""

import argparse, textwrap, pathlib

# ─── 1. Parse argumentos ───────────────────────────────────────────────────
p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
    description="Gera arquivos .cc/.h para TFLite-Micro a partir de um .tflite",
    epilog=textwrap.dedent("""
      Exemplos:
        python gerar_firmware_array.py --input modelo.tflite --name modelo_seno
        python gerar_firmware_array.py -i build.tflite -n my_model -o ../src
    """))
p.add_argument("-i","--input",   required=True,  help=".tflite de entrada")
p.add_argument("-n","--name",    required=True,  help="nome base do array")
p.add_argument("-o","--outdir",  default=".",    help="diretório de saída")
args = p.parse_args()

tflite_path = pathlib.Path(args.input).expanduser()
out_dir     = pathlib.Path(args.outdir).expanduser()
out_dir.mkdir(parents=True, exist_ok=True)

array_name  = args.name

# ─── 2. Lê bytes do modelo ─────────────────────────────────────────────────
model_bytes = tflite_path.read_bytes()
model_len   = len(model_bytes)

# ─── 3. Constrói o .cc ─────────────────────────────────────────────────────
hex_lines = []
for i, b in enumerate(model_bytes):
    if i % 12 == 0:
        hex_lines.append("  ")
    hex_lines[-1] += f"0x{b:02x}, "

cc_body = "\n".join(hex_lines).rstrip(", ")

cc_text = f"""// Gerado automaticamente por gerar_firmware_array.py
#include <stdint.h>
#include <stddef.h>

alignas(16) const uint8_t {array_name}[] = {{
{cc_body}
}};

const unsigned int {array_name}_len = {model_len};
"""

(out_dir / f"{array_name}.cc").write_text(cc_text, encoding="utf-8")

# ─── 4. Constrói o .h ──────────────────────────────────────────────────────
header_guard = f"{array_name.upper()}_H_"

h_text = f"""#ifndef {header_guard}
#define {header_guard}

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {{
#endif

extern const uint8_t  {array_name}[];
extern const unsigned int {array_name}_len;

#ifdef __cplusplus
}}  // extern "C"
#endif

#endif  // {header_guard}
"""

(out_dir / f"{array_name}.h").write_text(h_text, encoding="utf-8")

print(f"✓ Gerados:\n  {out_dir / (array_name + '.cc')}\n  {out_dir / (array_name + '.h')}")
