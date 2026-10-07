import argparse
import itertools
import json
import logging
import os
import random
import sys
import textwrap

from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Optional, TextIO, TypeVar, Union, cast

from . import VERSION, Faker, documentor, exceptions
from .chunked import CHUNKED_FORMATS, ChunkedProduction, ChunkSpec
from .config import AVAILABLE_LOCALES, DEFAULT_LOCALE, META_PROVIDERS_MODULES
from .documentor import Documentor
from .providers import BaseProvider

__author__ = "joke2k"

T = TypeVar("T")


def _encode_for_output(value: str, output: TextIO) -> str:
    encoding = getattr(output, "encoding", None)
    if encoding is None:
        return value

    try:
        value.encode(encoding)
    except UnicodeEncodeError:
        return value.encode(encoding, errors="backslashreplace").decode(encoding)

    return value


def print_provider(
    doc: Documentor,
    provider: BaseProvider,
    formatters: Dict[str, T],
    excludes: Optional[List[str]] = None,
    output: Optional[TextIO] = None,
) -> None:
    if output is None:
        output = sys.stdout
    if excludes is None:
        excludes = []

    print(file=output)
    print(_encode_for_output(f"### {doc.get_provider_name(provider)}", output), file=output)
    print(file=output)

    margin = max(30, doc.max_name_len + 2)
    for signature, example in formatters.items():
        if signature in excludes:
            continue
        signature_lines = textwrap.wrap(signature, width=margin, subsequent_indent="  ")
        try:
            lines = textwrap.wrap(
                str(example).expandtabs(),
                width=150 - margin,
                initial_indent="# ",
                subsequent_indent="  ",
            )
        except UnicodeDecodeError:
            # The example is actually made of bytes.
            # We could coerce to bytes, but that would fail anyway when we will
            # try to `print` the line.
            lines = ["<bytes>"]
        except UnicodeEncodeError:
            raise Exception(f"error on {signature!r} with value {example!r}")
        for left, right in itertools.zip_longest(signature_lines, lines, fillvalue=""):
            line = f"\t{left:<{margin}}  {right}"
            print(_encode_for_output(line, output), file=output)


def print_doc(
    provider_or_field: Optional[str] = None,
    args: Optional[List[T]] = None,
    lang: str = DEFAULT_LOCALE,
    output: Optional[Union[TextIO, TextIOWrapper]] = None,
    seed: Optional[float] = None,
    includes: Optional[List[str]] = None,
) -> None:
    if args is None:
        args = []
    if output is None:
        output = sys.stdout
    fake = Faker(locale=lang, includes=includes)
    fake.seed_instance(seed)

    from faker.providers import BaseProvider

    base_provider_formatters = list(dir(BaseProvider))

    if provider_or_field:
        if "." in provider_or_field:
            parts = provider_or_field.split(".")
            locale = parts[-2] if parts[-2] in AVAILABLE_LOCALES else lang
            fake = Faker(locale, providers=[provider_or_field], includes=includes)
            fake.seed_instance(seed)
            doc = documentor.Documentor(fake)
            doc.already_generated = base_provider_formatters
            print_provider(
                doc,
                fake.get_providers()[0],
                doc.get_provider_formatters(fake.get_providers()[0]),
                output=output,
            )
        else:
            try:
                print(fake.format(provider_or_field, *args), end="", file=output)
            except AttributeError:
                raise ValueError(f'No faker found for "{provider_or_field}({args})"')

    else:
        doc = documentor.Documentor(fake)
        unsupported: List[str] = []

        while True:
            try:
                formatters = doc.get_formatters(with_args=True, with_defaults=True, excludes=unsupported)
            except exceptions.UnsupportedFeature as e:
                unsupported.append(e.name)
            else:
                break

        for provider, fakers in formatters:
            print_provider(doc, provider, fakers, output=output)


class Command:
    def __init__(self, argv: Optional[str] = None) -> None:
        self.argv = argv or sys.argv[:]
        self.prog_name = Path(self.argv[0]).name

    def execute(self) -> None:
        """
        Given the command-line arguments, this creates a parser appropriate
        to that command, and runs it.
        """

        # retrieve default language from system environment
        default_locale = os.environ.get("LANG", "en_US").split(".")[0]
        if default_locale not in AVAILABLE_LOCALES:
            default_locale = DEFAULT_LOCALE

        epilog = f"""supported locales:

  {', '.join(sorted(AVAILABLE_LOCALES))}

  Faker can take a locale as an optional argument, to return localized data. If
  no locale argument is specified, the factory falls back to the user's OS
  locale as long as it is supported by at least one of the providers.
     - for this user, the default locale is {default_locale}.

  If the optional argument locale and/or user's default locale is not available
  for the specified provider, the factory falls back to faker's default locale,
  which is {DEFAULT_LOCALE}.

examples:

  $ faker address
  968 Bahringer Garden Apt. 722
  Kristinaland, NJ 09890

  $ faker -l de_DE address
  Samira-Niemeier-Allee 56
  94812 Biedenkopf

  $ faker profile ssn,birthdate
  {{'ssn': u'628-10-1085', 'birthdate': '2008-03-29'}}

  $ faker -r=3 -s=";" name
  Willam Kertzmann;
  Josiah Maggio;
  Gayla Schmitt;

"""

        formatter_class = argparse.RawDescriptionHelpFormatter
        parser = argparse.ArgumentParser(
            prog=self.prog_name,
            description=f"{self.prog_name} version {VERSION}",
            epilog=epilog,
            formatter_class=formatter_class,
        )

        parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")

        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="show INFO logging events instead "
            "of CRITICAL, which is the default. These logging "
            "events provide insight into localization of "
            "specific providers.",
        )

        parser.add_argument(
            "-o",
            metavar="output",
            type=argparse.FileType("w"),
            default=sys.stdout,
            help="redirect output to a file",
        )

        parser.add_argument(
            "-l",
            "--lang",
            choices=AVAILABLE_LOCALES,
            default=default_locale,
            metavar="LOCALE",
            help="specify the language for a localized provider (e.g. de_DE)",
        )
        parser.add_argument(
            "-r",
            "--repeat",
            default=1,
            type=int,
            help="generate the specified number of outputs",
        )
        parser.add_argument(
            "-s",
            "--sep",
            default="\n",
            help="use the specified separator after each output",
        )

        parser.add_argument(
            "--seed",
            metavar="SEED",
            type=int,
            help="specify a seed for the random generator so "
            "that results are repeatable. Also compatible "
            "with 'repeat' option",
        )

        parser.add_argument(
            "--chunk-size",
            metavar="ROWS",
            type=int,
            default=None,
            help="stream the structured fake named by the positional argument "
            "(one of: {formats}) chunk by chunk, with ROWS rows per chunk. "
            "Each chunk is written and flushed immediately, and '-r/--repeat' "
            "sets the total number of rows.".format(formats=", ".join(CHUNKED_FORMATS)),
        )
        parser.add_argument(
            "--on-error",
            choices=["skip", "replace", "abort"],
            default="skip",
            help="row failure policy for chunked output: 'skip' omits the row, "
            "'replace' substitutes it with --replace-with, 'abort' invalidates "
            "the chunk (default: skip)",
        )
        parser.add_argument(
            "--replace-with",
            metavar="VALUE",
            default=None,
            help="substitute value for failing rows when --on-error=replace",
        )
        parser.add_argument(
            "--max-chunks",
            metavar="N",
            type=int,
            default=None,
            help="maximum number of chunks a chunked production may deliver",
        )
        parser.add_argument(
            "--max-failures",
            metavar="N",
            type=int,
            default=None,
            help="maximum number of row failures a chunked production may record",
        )
        parser.add_argument(
            "--failure-log",
            metavar="PATH",
            default=None,
            help="write the chunked-output failure list (row, column, definition, "
            "error) as JSON to PATH; the file is only created when failures occur",
        )

        parser.add_argument(
            "-i",
            "--include",
            action="append",
            help="list of additional custom providers to "
            "user, given as the import path of the module "
            "containing your Provider class (not the provider "
            "class itself)",
        )

        parser.add_argument(
            "fake",
            action="store",
            nargs="?",
            help="name of the fake to generate output for (e.g. profile)",
        )

        parser.add_argument(
            "fake_args",
            metavar="fake argument",
            action="store",
            nargs="*",
            help="optional arguments to pass to the fake "
            "(e.g. the profile fake takes an optional "
            "list of comma separated field names as the "
            "first argument)",
        )

        arguments = parser.parse_args(self.argv[1:])
        if arguments.include is None:
            arguments.include = META_PROVIDERS_MODULES

        if arguments.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.CRITICAL)

        if arguments.chunk_size is not None:
            if not arguments.fake:
                parser.error("chunked output requires a structured fake name: " + ", ".join(CHUNKED_FORMATS))
            if arguments.fake not in CHUNKED_FORMATS:
                parser.error(
                    f"chunked output only supports {', '.join(CHUNKED_FORMATS)}, not {arguments.fake!r}",
                )
            if arguments.fake_args:
                parser.error("chunked output does not accept positional fake arguments")
            self._execute_chunked(arguments)
            return

        random.seed(arguments.seed)
        seeds = [random.random() for _ in range(arguments.repeat)]

        for i in range(arguments.repeat):
            print_doc(
                arguments.fake,
                arguments.fake_args,
                lang=arguments.lang,
                output=arguments.o,
                seed=seeds[i],
                includes=arguments.include,
            )
            print(arguments.sep, file=arguments.o)

            if not arguments.fake:
                # repeat not supported for all docs
                break

    def _execute_chunked(self, arguments: argparse.Namespace) -> None:
        """Stream one structured fake in chunks instead of a one-shot loop.

        Every chunk is written and flushed as soon as it is delivered, so the
        output always ends in a well-defined state, and the accumulated row
        failure list can be written to ``--failure-log``.
        """
        spec_kwargs = {
            "rows_per_chunk": arguments.chunk_size,
            "failure_policy": "abort_chunk" if arguments.on_error == "abort" else arguments.on_error,
            "replacement": arguments.replace_with,
        }
        if arguments.max_chunks is not None:
            spec_kwargs["max_chunks"] = arguments.max_chunks
        if arguments.max_failures is not None:
            spec_kwargs["max_failures"] = arguments.max_failures

        output: TextIO = arguments.o
        delivered = 0
        invalidated = 0
        session = None
        rejected = False

        try:
            spec = ChunkSpec(**spec_kwargs)
            fake = Faker(locale=arguments.lang, includes=arguments.include)
            fake.seed_instance(arguments.seed)
            session = fake.produce_chunks(
                arguments.fake,
                spec,
                num_rows=arguments.repeat,
            )
            for chunk in session:
                payload = chunk.payload
                if isinstance(payload, bytes):
                    payload = payload.decode()
                output.write(payload)
                output.write(arguments.sep)
                output.flush()
                delivered += 1
                if chunk.status.value == "invalidated":
                    invalidated += 1
        except exceptions.ChunkedProductionError as exc:
            rejected = True
            print(f"chunked output rejected: {exc}", file=sys.stderr)
        finally:
            failure_count = self._finalize_chunked_output(output, delivered, session, arguments.failure_log)

        if failure_count:
            print(
                f"chunked output finished with {failure_count} row failure(s) "
                f"handled by policy {arguments.on_error!r}"
                + (f", {invalidated} chunk(s) invalidated" if invalidated else ""),
                file=sys.stderr,
            )

        if rejected:
            raise SystemExit(1)

    @staticmethod
    def _finalize_chunked_output(
        output: TextIO,
        delivered: int,
        session: Optional[ChunkedProduction],
        failure_log_path: Optional[str],
    ) -> int:
        """Flush/close out the chunked run and persist the failure list.

        Returns the number of accumulated row failures.  When nothing was
        delivered to a real output file, the (already truncated) file is
        removed so a rejected run leaves no empty artifact behind.
        """
        failures = list(session.failures) if session is not None else []

        if failure_log_path and failures:
            with open(failure_log_path, "w", encoding="utf-8") as failure_log:
                json.dump([failure.to_dict() for failure in failures], failure_log, indent=2)

        output_path = cast(Optional[str], getattr(output, "name", None))
        is_real_file = bool(output_path) and output not in (sys.stdout, sys.stderr)

        try:
            output.flush()
        except (OSError, ValueError):
            pass

        if delivered == 0 and is_real_file and output_path is not None:
            # argparse opened (and truncated) the file before production
            # started; remove it if the run never delivered anything.
            try:
                output.close()
            except (OSError, ValueError):
                pass
            try:
                if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                    os.remove(output_path)
            except OSError:
                pass

        return len(failures)


def execute_from_command_line(argv: Optional[str] = None) -> None:
    """A simple method that runs a Command."""
    if sys.stdout.encoding is None:
        print(
            "please set python env PYTHONIOENCODING=UTF-8, example: "
            "export PYTHONIOENCODING=UTF-8, when writing to stdout",
            file=sys.stderr,
        )
        exit(1)

    command = Command(argv)
    command.execute()


if __name__ == "__main__":
    execute_from_command_line()
