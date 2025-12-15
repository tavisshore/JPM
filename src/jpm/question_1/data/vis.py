
from colorama import Back, Fore, Style, init

# Initialize colorama
init(autoreset=True)


def pretty_print_mapping(mapping_dict, indent=0):
    """
    Pretty print the feature mapping dictionary with colors.

    Parameters:
    -----------
    mapping_dict : dict
        Dictionary mapping new features to lists of old features
    indent : int
        Current indentation level (used for recursion)
    """

    indent_str = "  " * indent

    for key, value in mapping_dict.items():
        if key == "__unmapped__":
            # Special handling for unmapped
            if isinstance(value, list) and value:
                print(f"{indent_str}{Fore.YELLOW}⚠ {key}:{Style.RESET_ALL}")
                for item in value:
                    print(f"{indent_str}  {Fore.YELLOW}• {item}{Style.RESET_ALL}")
            elif isinstance(value, list):
                print(f"{indent_str}{Fore.YELLOW}⚠ {key}: []{Style.RESET_ALL}")
            continue

        if isinstance(value, dict):
            # Section header (like "Assets", "Current", etc.)
            if indent == 0:
                print(f"\n{Fore.CYAN}{Back.BLACK}{'='*80}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{Back.BLACK} {key.upper()} {Style.RESET_ALL}")
                print(f"{Fore.CYAN}{Back.BLACK}{'='*80}{Style.RESET_ALL}\n")
            else:
                print(f"{indent_str}{Fore.MAGENTA}▼ {key}{Style.RESET_ALL}")

            # Recurse into nested dictionary
            pretty_print_mapping(value, indent + 1)

        elif isinstance(value, list):
            # Leaf node - show mapping
            if value:
                # Has mappings
                print(f"{indent_str}{Fore.GREEN}✓ {key}:{Style.RESET_ALL}")
                for old_feature in value:
                    print(
                        f"{indent_str}  {Fore.WHITE}├─ {old_feature}{Style.RESET_ALL}"
                    )
            else:
                # Empty list - no mappings
                print(
                    f"{indent_str}{Fore.RED}✗ {key}: {Fore.LIGHTBLACK_EX}[no mapping]{Style.RESET_ALL}"
                )


def print_mapping_summary(mapping_dict):
    """
    Print a summary of the mapping statistics.

    Parameters:
    -----------
    mapping_dict : dict
        Dictionary mapping new features to lists of old features
    """

    def count_mappings(d):
        """Recursively count features."""
        mapped_count = 0
        unmapped_count = 0
        total_old_features = set()

        for key, value in d.items():
            if key == "__unmapped__":
                if isinstance(value, list):
                    total_old_features.update(value)
                continue

            if isinstance(value, dict):
                m, u, old = count_mappings(value)
                mapped_count += m
                unmapped_count += u
                total_old_features.update(old)
            elif isinstance(value, list):
                if value:
                    mapped_count += 1
                    total_old_features.update(value)
                else:
                    unmapped_count += 1

        return mapped_count, unmapped_count, total_old_features

    mapped, unmapped, old_features = count_mappings(mapping_dict)
    total_new = mapped + unmapped

    print(f"\n{Fore.CYAN}{Back.BLACK}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Back.BLACK} MAPPING SUMMARY {Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Back.BLACK}{'='*80}{Style.RESET_ALL}\n")

    print(f"{Fore.WHITE}Total New Features: {Fore.CYAN}{total_new}{Style.RESET_ALL}")
    print(
        f"{Fore.GREEN}  ✓ Mapped: {mapped} ({100*mapped/total_new:.1f}%){Style.RESET_ALL}"
    )
    print(
        f"{Fore.RED}  ✗ Unmapped: {unmapped} ({100*unmapped/total_new:.1f}%){Style.RESET_ALL}"
    )
    print(
        f"{Fore.WHITE}Total Old Features Used: {Fore.CYAN}{len(old_features)}{Style.RESET_ALL}"
    )
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")


def pretty_print_full_mapping(mapping_dict, show_summary=True):
    """
    Complete pretty print with optional summary.

    Parameters:
    -----------
    mapping_dict : dict
        Dictionary mapping new features to lists of old features
    show_summary : bool
        Whether to show summary statistics at the end
    """

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'#'*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}#{' '*78}#{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}{Style.BRIGHT}#  FEATURE MAPPING: OLD → NEW{' '*48}#{Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{Style.BRIGHT}#{' '*78}#{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}{'#'*80}{Style.RESET_ALL}")

    pretty_print_mapping(mapping_dict)

    if show_summary:
        print_mapping_summary(mapping_dict)
